// Experimental layout only. Keep original shader source and arithmetic intact.
#include "rife.hlsl"
Buffer<float4> packedFlow:register(t5);
Buffer<float> packedMask:register(t6);
Buffer<uint> rawMotion:register(t7);
RWBuffer<uint4> flowOutput:register(u1);
RWBuffer<uint> maskOutput:register(u2);
[numthreads(256,1,1)]
void packMotion(uint i:SV_DispatchThreadID) {
    uint n=guideW*guideH;
    if(i<n){flowOutput[i]=uint4(rawMotion[i],rawMotion[n+i],rawMotion[2*n+i],rawMotion[3*n+i]);maskOutput[i]=rawMotion[4*n+i];}
}
float2 sampleBranchPacked(float2 p,int2 j,uint outputPlane,out float valid) {
    uint i=j.y*guideW+j.x,n=guideW*guideH;
    float4 v=packedFlow[i];
    float m=saturate(packedMask[i]);
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
float4 compositePlanePacked(float4 screen,uint outputPlane) {
    float2 p=screen.xy*(outputPlane?2.0:1.0);
    float2 g=sourceToGuide(p)-0.5,f=frac(g);
    int2 lo=int2(floor(g)),hi=int2(guideW,guideH)-1;
    float2 c=0;float total=0;
    [unroll]for(int y=0;y<2;++y)[unroll]for(int x=0;x<2;++x){
        int2 j=clamp(lo+int2(x,y),0,hi);float valid;
        float2 v=sampleBranchPacked(p,j,outputPlane,valid);
        float weight=(x?f.x:1-f.x)*(y?f.y:1-f.y)*valid;
        c+=v*weight;total+=weight;
    }
    c=total>0.001?c/total:color(p,false,outputPlane);
    if(tenBit)c=clamp(round(c*65535.0/64.0),0,1023)*64.0/65535.0;
    return float4(c,0,1);
}

float4 compositePackedY(float4 screen:SV_Position):SV_Target {return compositePlanePacked(screen,0);}
float4 compositePackedUV(float4 screen:SV_Position):SV_Target {return compositePlanePacked(screen,1);}
