Texture2D<float> sourceA:register(t0);Texture2D<float> sourceB:register(t1);
Texture2D<float2> chromaA:register(t2);Texture2D<float2> chromaB:register(t3);
RWStructuredBuffer<uint> stats:register(u0);
cbuffer Dimensions:register(b0){uint width;uint height;uint2 padding;};
// 80 aggregate DWORDs, one identity flag, then a GPU-only packed 64x36 grid.
uint2 sampleGrid(uint2 id){uint2 v=0;
 [unroll]for(uint oy=1;oy<=3;oy+=2)[unroll]for(uint ox=1;ox<=3;ox+=2){
  int3 p=int3((id.x*4+ox)*width/256,(id.y*4+oy)*height/144,0);
  v+=uint2(round(sourceA.Load(p)*255),round(sourceB.Load(p)*255));
 }return (v+2)/4;
}
void accumulateGrid(uint2 id){
 if(id.x>=64||id.y>=36)return;
 uint2 v=sampleGrid(id.xy);uint a=v.x,b=v.y;int delta=int(b)-int(a);uint diff=uint(abs(delta));
 stats[81+id.y*64+id.x]=a|(b<<8);
 InterlockedAdd(stats[0],1);InterlockedAdd(stats[1],diff);InterlockedAdd(stats[2],asuint(delta));InterlockedAdd(stats[3],uint(delta*delta));
 if(diff>=32)InterlockedAdd(stats[4],1);InterlockedAdd(stats[5+(a>>3)],1);InterlockedAdd(stats[37+(b>>3)],1);
 InterlockedAdd(stats[69],a);InterlockedAdd(stats[70],b);InterlockedAdd(stats[71],a*a);InterlockedAdd(stats[72],b*b);InterlockedAdd(stats[73],a*b);
 uint ea=0,eb=0;int ep=0;
 if(id.x<63){int2 d=int2(sampleGrid(id.xy+uint2(1,0)))-int2(v);ea+=d.x*d.x;eb+=d.y*d.y;ep+=d.x*d.y;}
 if(id.y<35){int2 d=int2(sampleGrid(id.xy+uint2(0,1)))-int2(v);ea+=d.x*d.x;eb+=d.y*d.y;ep+=d.x*d.y;}
 InterlockedAdd(stats[74],ea);InterlockedAdd(stats[75],eb);InterlockedAdd(stats[76],asuint(ep));
}

// Preserve the full-resolution reference entry for offline parity tests.
groupshared uint different;
[numthreads(16,16,1)]void main(uint3 id:SV_DispatchThreadID,uint tid:SV_GroupIndex){
 if(tid==0)different=0;GroupMemoryBarrierWithGroupSync();bool changed=false;
 if(id.x<width&&id.y<height){int3 p=int3(id.xy,0);changed=sourceA.Load(p)!=sourceB.Load(p);}
 if(id.x<width/2&&id.y<height/2){int3 p=int3(id.xy,0);changed=changed||any(chromaA.Load(p)!=chromaB.Load(p));}
 if(changed)InterlockedOr(different,1);GroupMemoryBarrierWithGroupSync();if(tid==0&&different)InterlockedOr(stats[80],1);

 accumulateGrid(id.xy);
}
// A raw (unquantized) difference proves that the frames are not identical.
// A matching small grid proves nothing: identity() then checks EVERY Y/UV pixel.
[numthreads(16,16,1)]void gridProbe(uint3 id:SV_DispatchThreadID){
 if(id.x>=64||id.y>=36)return;
 accumulateGrid(id.xy);bool changed=false;
 [unroll]for(uint oy=1;oy<=3;oy+=2)[unroll]for(uint ox=1;ox<=3;ox+=2){
  int3 p=int3((id.x*4+ox)*width/256,(id.y*4+oy)*height/144,0);
  changed=changed||(sourceA.Load(p)!=sourceB.Load(p));
  p.xy/=2;changed=changed||any(chromaA.Load(p)!=chromaB.Load(p));
 }
 if(changed)InterlockedOr(stats[80],1);
}
// Bound dispatch overhead even at 8K. D3D11 orders the preceding grid UAV
// writes before this dispatch. No CPU readback, extra flush or wait is added.
groupshared uint probeDifferent;
[numthreads(256,1,1)]void identity(uint3 id:SV_DispatchThreadID,uint tid:SV_GroupIndex){
 if(tid==0){InterlockedOr(stats[80],0,probeDifferent);different=0;}
 GroupMemoryBarrierWithGroupSync();if(probeDifferent)return;
 bool changed=false;
 // Compute quotient/remainder once per thread, not for each 8K pixel.
 uint2 q=uint2(id.x%width,id.x/width),step=uint2(1048576%width,1048576/width);
 [loop]while(q.y<height){
  if(sourceA.Load(int3(q,0))!=sourceB.Load(int3(q,0))){changed=true;break;}
  q+=step;if(q.x>=width){q.x-=width;++q.y;}
 }
 uint cw=width/2;q=uint2(id.x%cw,id.x/cw);step=uint2(1048576%cw,1048576/cw);
 [loop]while(!changed&&q.y<height/2){
  if(any(chromaA.Load(int3(q,0))!=chromaB.Load(int3(q,0)))){changed=true;break;}
  q+=step;if(q.x>=cw){q.x-=cw;++q.y;}
 }
 if(changed)InterlockedOr(different,1);
 GroupMemoryBarrierWithGroupSync();if(tid==0&&different)InterlockedOr(stats[80],1);
}
float grid(int x,int y,uint direction){uint packed=stats[81+y*64+x];return direction?(packed>>8)&255:packed&255;}
groupshared uint eligible,patchWeight;
groupshared float patchSum,patchEnergy;
groupshared float maxima[64];
[numthreads(64,1,1)]void match(uint3 group:SV_GroupID,uint tid:SV_GroupIndex){
 uint direction=group.x/40;int x=(group.x%8)*8,y=((group.x%40)/8)*6;
 if(tid==0){
  float n=stats[0],ma=stats[69]/n,mb=stats[70]/n;
  float va=max(0.,stats[71]/n-ma*ma),vb=max(0.,stats[72]/n-mb*mb);
  float norm=(stats[1]/n)/max(8.,sqrt((va+vb)*.5));
  float edge=asint(stats[76])/max(1.,sqrt(float(stats[74])*float(stats[75])));
  float corr=(stats[73]/n-ma*mb)/max(1.,sqrt(va*vb));float hist=0.;
  uint first=0,second=0,secondCount=0;
  [loop]for(uint k=0;k<32;++k){hist+=abs(float(stats[5+k])-float(stats[37+k]));if(stats[5+k]>stats[5+first])first=k;}hist/=2*n;
  [loop]for(uint k=0;k<32;++k)if(abs(int(k)-int(first))>=12&&stats[5+k]>=secondCount){second=k;secondCount=stats[5+k];}
  uint a0=0,a1=0,b0=0,b1=0;
  [loop]for(uint k=0;k<32;++k){if(abs(int(k)-int(first))<=2){a0+=stats[5+k];b0+=stats[37+k];}if(abs(int(k)-int(second))<=2){a1+=stats[5+k];b1+=stats[37+k];}}
  float palette=min(min(a0,a1),min(b0,b1))>=.05*n?min(a0+a1,b0+b1)/n:0.;
  eligible=stats[1]/n>=6.&&norm>=.65&&hist>=.10&&edge<.35&&corr<.90&&!(palette>=.80&&norm<2.)&&!(edge>=.25&&corr>=.60);
  float sum=0.,square=0.;
  [loop]for(int j=0;j<7;++j)[loop]for(int i=0;i<7;++i){float v=grid(x+i,y+j,direction);sum+=v;square+=v*v;}
  patchSum=sum;patchEnergy=max(0.,square-sum*sum/49.);float sd=sqrt(patchEnergy/49.);
  eligible=eligible&&(sd>=max(3.,.12*sqrt(direction?vb:va)));patchWeight=uint(round(min(sd,30.)*256.));
 }
 GroupMemoryBarrierWithGroupSync();if(!eligible)return;
 int xmin=max(0,x-10),xmax=min(57,x+10),ymin=max(0,y-6),ymax=min(29,y+6),columns=xmax-xmin+1;
 float best=-1.;
 [loop]for(uint k=tid;k<uint(columns*(ymax-ymin+1));k+=64){int xx=xmin+k%columns,yy=ymin+k/columns;float sb=0.,sbb=0.,sab=0.;
  [loop]for(int j=0;j<7;++j)[loop]for(int i=0;i<7;++i){float a=grid(x+i,y+j,direction),b=grid(xx+i,yy+j,1-direction);sb+=b;sbb+=b*b;sab+=a*b;}
  float energy=sbb-sb*sb/49.;float correlation=energy>1e-5?(sab-patchSum*sb/49.)*rsqrt(patchEnergy*energy):0.;best=max(best,min(1.,correlation));
 }
 maxima[tid]=best;GroupMemoryBarrierWithGroupSync();
 [unroll]for(uint stride=32;stride;stride>>=1){if(tid<stride)maxima[tid]=max(maxima[tid],maxima[tid+stride]);GroupMemoryBarrierWithGroupSync();}
 if(tid==0){InterlockedAdd(stats[77],patchWeight);InterlockedAdd(stats[78],asuint(int(round(maxima[0]*patchWeight*1024.))));if(maxima[0]>.80)InterlockedAdd(stats[79],patchWeight);}
}
