// RIFE guide only; final colors always come from native NV12/P010 source planes.
Texture2D<float> ay:register(t0);
Texture2D<float> by:register(t1);
Texture2D<float2> auv:register(t2);
Texture2D<float2> buv:register(t3);
Buffer<float> motion:register(t4);
RWBuffer<float> input:register(u0);
SamplerState linearClamp:register(s0);
cbuffer Parameters:register(b0) {
    float2 size; uint guideW,guideH;
    float phase; uint plane; uint tenBit; uint hdrTransfer;
    float2 active; float2 offset;
    uint portrait; float3 unused;
};
float4 vs(uint v:SV_VertexID):SV_Position {
    return float4(v==0?float2(-1,-1):(v==1?float2(-1,3):float2(3,-1)),0,1);
}
float2 guideToSource(float2 g) {
    float2 q=(g-offset)/active;
    if(portrait)q=float2(q.y,1.0-q.x);
    return q*size;
}
float2 sourceToGuide(float2 p) {
    float2 q=p/size;
    if(portrait)q=float2(1.0-q.y,q.x);
    return q*active+offset;
}
float yAt(float2 p,bool next) {
    return next?by.SampleLevel(linearClamp,p/size,0):ay.SampleLevel(linearClamp,p/size,0);
}
float2 uvAt(float2 p,bool next) {
    return next?buv.SampleLevel(linearClamp,p/size,0):auv.SampleLevel(linearClamp,p/size,0);
}
float2 color(float2 p,bool next,uint outputPlane) {return outputPlane?uvAt(p,next):float2(yAt(p,next),0);}
float3 rgbAt(float2 p,bool next) {
    float scale=tenBit?65535.0/65280.0:1.0;
    float y=(yAt(p,next)*scale*255.0-16.0)/219.0;
    float2 uv=(uvAt(p,next)*scale*255.0-128.0)/224.0;
    // PQ/HLG stay in their nonlinear domain. No tone mapping of output pixels.
    if(hdrTransfer)return saturate(float3(y+1.4746*uv.y,y-0.164553*uv.x-0.571353*uv.y,y+1.8814*uv.x));
    return saturate(float3(y+1.5748*uv.y,y-0.187324*uv.x-0.468124*uv.y,y+1.8556*uv.x));
}
[numthreads(16,16,1)]
void prepareInput(uint3 id:SV_DispatchThreadID) {
    if(id.x>=guideW||id.y>=guideH)return;
    uint i=id.y*guideW+id.x,n=guideW*guideH;
    float2 p=guideToSource(float2(id.xy)+0.5);
    float3 a=rgbAt(p,false),b=rgbAt(p,true);
    input[i]=a.r;input[n+i]=a.g;input[2*n+i]=a.b;
    input[3*n+i]=b.r;input[4*n+i]=b.g;input[5*n+i]=b.b;
    input[6*n+i]=phase;
}
[numthreads(16,16,1)]
void setTime(uint3 id:SV_DispatchThreadID) {
    if(id.x<guideW&&id.y<guideH)input[6*guideW*guideH+id.y*guideW+id.x]=phase;
}
float2 sampleBranch(float2 p,int2 j,uint outputPlane,out float valid) {
    uint i=j.y*guideW+j.x,n=guideW*guideH;
    float4 v=float4(motion[i],motion[n+i],motion[2*n+i],motion[3*n+i]);
    float m=saturate(motion[4*n+i]);
    if(!all(isfinite(v))||!isfinite(m)){valid=0;return 0;}
    float2 g=sourceToGuide(p);
    float2 a=g+v.xy-0.5,b=g+v.zw-0.5;
    float2 extent=float2(guideW,guideH)-1.0;
    float2 padded=ceil(float2(guideW,guideH)/32.0)*32.0-1.0;
    a=clamp(a,0.0,padded);b=clamp(b,0.0,padded);
    a=min(a,2.0*extent-a)+0.5;b=min(b,2.0*extent-b)+0.5;
    valid=1;
    return color(guideToSource(a),false,outputPlane)*m+color(guideToSource(b),true,outputPlane)*(1.0-m);
}
float4 compositePlane(float4 screen,uint outputPlane) {
    float2 p=screen.xy*(outputPlane?2.0:1.0);
    float2 g=sourceToGuide(p)-0.5,f=frac(g);
    int2 lo=int2(floor(g)),hi=int2(guideW,guideH)-1;
    float2 c=0;float total=0;
    [unroll]for(int y=0;y<2;++y)[unroll]for(int x=0;x<2;++x){
        int2 j=clamp(lo+int2(x,y),0,hi);float valid;
        float2 v=sampleBranch(p,j,outputPlane,valid);
        float weight=(x?f.x:1-f.x)*(y?f.y:1-f.y)*valid;
        c+=v*weight;total+=weight;
    }
    c=total>0.001?c/total:color(p,false,outputPlane);
    if(tenBit)c=clamp(round(c*65535.0/64.0),0,1023)*64.0/65535.0;
    return float4(c,0,1);
}

// Compile both native planes ahead of playback. Specialization removes unused
// source texture fetches/branches without changing the four motion branches.
float4 composite(float4 screen:SV_Position):SV_Target {return compositePlane(screen,plane);}
float4 compositeY(float4 screen:SV_Position):SV_Target {return compositePlane(screen,0);}
float4 compositeUV(float4 screen:SV_Position):SV_Target {return compositePlane(screen,1);}
