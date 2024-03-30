namespace shaders {
const char vs_shader[] =
    "cbuffer PerFaceConstants : register (b0)\n"
    "{\n"
    "	matrix MVP;\n"
    "	uint isWallDestroyed[1];\n"
    "}\n"
    "struct VertexShaderOutput\n"
    "{\n"
    "	float4 position : SV_POSITION;\n"
    "	float2 uv : TEXCOORD;\n"
    "};\n"
    "VertexShaderOutput VS_main(\n"
    "	uint extrudVertID : EXTRUDVERTID,\n"
    "	float2 uv : TEXCOORD,\n" 
    "   matrix instanceExtrudingVer : INSTANCEEXTRUDINGVERTS,\n"
    "	uint id: SV_InstanceID)\n"
    "{\n"
    "	VertexShaderOutput output;\n"
    "   if (isWallDestroyed[id] == 0) {\n"
    "   float4 extruded = instanceExtrudingVer[extrudVertID];\n"
    "   output.position = mul(MVP, extruded);\n"
    "	output.uv = uv;\n"
    "	} else {\n"
    "	/*primitive discarding by degradating triangle with putting outside NDC */\n"
    "	output.position = float4(100, 100, 100, 1);\n"
    "	}\n"
    "	return output;\n"
    "}\n";
const char fs_shader[] =
    "Texture2D<float4> inputTexture : register(t0);\n"
    "SamplerState     texureSampler : register(s0);\n"
    "float4 PS_main (float4 position : SV_POSITION,\n"
    "				float2 uv : TEXCOORD) : SV_TARGET\n"
    "{\n"
    "	return inputTexture.Sample (texureSampler, uv);\n"
    "}\n";
}
