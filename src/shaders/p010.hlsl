float4 vs(uint vertex:SV_VertexID):SV_Position {
 float2 position=vertex==0?float2(-1,-1):(vertex==1?float2(-1,3):float2(3,-1));
 return float4(position,0,1);
}

Texture2D<uint4> source:register(t0);
float4 ps(float4 position:SV_Position):SV_Target {
 uint4 code=source.Load(int3(int2(position.xy),0))>>6;
 return float4(code << 6)/65535.0f;
}
