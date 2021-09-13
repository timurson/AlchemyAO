
-- Vertex

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoords;

out vec2 TexCoords;

void main()
{
    TexCoords = aTexCoords;
    gl_Position = vec4(aPos, 1.0);
}

-- Fragment

out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gDiffuse;
uniform sampler2D gSpecular;
uniform sampler2D shadowMap;
uniform sampler2D ambientOcclusion;
uniform mat4 lightSpaceMatrix;
uniform float shadowSaturation;

// IBL
uniform samplerCube environmentMap;
uniform samplerCube irradianceMap;
uniform sampler2D brdfLUT;

struct Light {
    vec3 Position;
    vec3 Color;
	float Radius;
	float Intensity;
};

uniform Light gLight;
uniform vec3 viewPos;
uniform int shadowMethod; // 0 - standard, 1 - MSM
uniform int iblSamples;

const float PI = 3.14159265359;

// ----------------------------------------------------------------------------
float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a = roughness*roughness;
    float a2 = a*a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}

// ----------------------------------------------------------------------------
float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}

// ----------------------------------------------------------------------------
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

// ----------------------------------------------------------------------------
vec3 FresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

// ----------------------------------------------------------------------------
vec3 FresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(1.0 - cosTheta, 5.0);
}   

// ----------------------------------------------------------------------------
// http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
// efficient VanDerCorpus calculation.
float RadicalInverse_VdC(uint bits) 
{
     bits = (bits << 16u) | (bits >> 16u);
     bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
     bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
     bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
     bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
     return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}
// ----------------------------------------------------------------------------
vec2 Hammersley(uint i, uint N)
{
	return vec2(float(i)/float(N), RadicalInverse_VdC(i));
}

// ----------------------------------------------------------------------------
vec3 ImportanceSampleGGX(vec2 Xi, vec3 N, float roughness)
{
	float a = roughness*roughness;
	
	float phi = 2.0 * PI * Xi.x;
	float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a*a - 1.0) * Xi.y));
	float sinTheta = sqrt(1.0 - cosTheta*cosTheta);
	
	// from spherical coordinates to cartesian coordinates - halfway vector
	vec3 H;
	H.x = cos(phi) * sinTheta;
	H.y = sin(phi) * sinTheta;
	H.z = cosTheta;
	
	// from tangent-space H vector to world-space sample vector
	vec3 up          = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
	vec3 tangent   = normalize(cross(up, N));
	vec3 bitangent = cross(N, tangent);
	
	vec3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
	return normalize(sampleVec);
}

// ----------------------------------------------------------------------------
float CalculateShadow(vec3 fragPos, vec3 normal)
{
	vec4 fragPosLightSpace = lightSpaceMatrix * vec4(fragPos, 1.0);
	// perform perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
	// transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;
	// calculate bias
	vec3 lightDir = normalize(gLight.Position - fragPos);
	float bias = max(0.01 * (1.0 - dot(normal, lightDir)), 0.005);
	
	float shadowDepth = texture(shadowMap, projCoords.xy).r; 
	
	float shadowCoef = projCoords.z - bias > shadowDepth  ? 0.0 : 1.0;
	
	// keep the shadow at 1.0 when outside the zFar region of the light's frustum.
    if(projCoords.z > 1.0)
        shadowCoef = 1.0;
	
	return shadowCoef;
}

float Linstep(float min, float max, float v)  
{
	return clamp ((v - min) / (max - min), 0.0, 1.0);
}  

float ReduceLightBleeding(float p_max, float amount)  
{  
   return Linstep(amount, 1, p_max);  
}  

float CalculateMSMHamburger(vec4 moments, float frag_depth, float depth_bias, float moment_bias)
{
	// Bias input data to avoid artifacts
    vec4 b = mix(moments, vec4(0.5f, 0.5f, 0.5f, 0.5f), moment_bias);
	vec3 z;
    z[0] = frag_depth - depth_bias;

    // Compute a Cholesky factorization of the Hankel matrix B storing only non-
    // trivial entries or related products
    float L32D22 = fma(-b[0], b[1], b[2]);
    float D22 = fma(-b[0], b[0], b[1]);
    float squaredDepthVariance = fma(-b[1], b[1], b[3]);
    float D33D22 = dot(vec2(squaredDepthVariance, -L32D22), vec2(D22, L32D22));
    float InvD22 = 1.0f / D22;
    float L32 = L32D22 * InvD22;

    // Obtain a scaled inverse image of bz = (1,z[0],z[0]*z[0])^T
    vec3 c = vec3(1.0f, z[0], z[0] * z[0]);

    // Forward substitution to solve L*c1=bz
    c[1] -= b.x;
    c[2] -= b.y + L32 * c[1];

    // Scaling to solve D*c2=c1
    c[1] *= InvD22;
    c[2] *= D22 / D33D22;

    // Backward substitution to solve L^T*c3=c2
    c[1] -= L32 * c[2];
    c[0] -= dot(c.yz, b.xy);

    // Solve the quadratic equation c[0]+c[1]*z+c[2]*z^2 to obtain solutions
    // z[1] and z[2]
    float p = c[1] / c[2];
    float q = c[0] / c[2];
    float D = (p * p * 0.25f) - q;
    float r = sqrt(D);
    z[1] =- p * 0.5f - r;
    z[2] =- p * 0.5f + r;

    // Compute the shadow intensity by summing the appropriate weights
    vec4 switchVal = (z[2] < z[0]) ? vec4(z[1], z[0], 1.0f, 1.0f) :
                      ((z[1] < z[0]) ? vec4(z[0], z[1], 0.0f, 1.0f) :
                      vec4(0.0f,0.0f,0.0f,0.0f));
    float quotient = (switchVal[0] * z[2] - b[0] * (switchVal[0] + z[2]) + b[1])/((z[2] - switchVal[1]) * (z[0] - z[1]));
    float shadowIntensity = switchVal[2] + switchVal[3] * quotient;
    return 1.0f - clamp(shadowIntensity, 0.0f, 1.0f);
}

float CalculateShadow4MSM(vec3 fragPos)
{
	vec4 fragPosLightSpace = lightSpaceMatrix * vec4(fragPos, 1.0);
	// perform perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
	// transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;
	// keep the shadow at 1.0 when outside the zFar region of the light's frustum.
    if(projCoords.z > 1.0)
        return 1.0;
	
	float currentDepth = projCoords.z;
	
	return CalculateMSMHamburger(texture(shadowMap, projCoords.xy), currentDepth, 0.0000, 0.0003);
}

// ----------------------------------------------------------------------------
vec3 SpecularIBL(vec3 N, vec3 V, float roughness)
{
	vec3 specularLighting = vec3(0.0);
	float totalWeight = 0.0;
	
	for(uint i = 0u; i < iblSamples; ++i)
	{
		// generates a sample vector that's biased towards the preferred alignment direction (importance sampling).
        vec2 Xi = Hammersley(i, iblSamples);
		vec3 H = ImportanceSampleGGX(Xi, N, roughness);
		vec3 L  = normalize(2.0 * dot(V, H) * H - V);
		float NdotL = max(dot(N, L), 0.0);
		if(NdotL > 0.0)
        {
			// sample from the environment's mip level based on roughness/pdf
            float D   = DistributionGGX(N, H, roughness);
            float NdotH = max(dot(N, H), 0.0);
            float HdotV = max(dot(H, V), 0.0);
            float pdf = D * NdotH / (4.0 * HdotV) + 0.0001; 
			
			float resolution = 512.0; // resolution of source cubemap (per face)
			float saTexel  = 4.0 * PI / (6.0 * resolution * resolution);
            float saSample = 1.0 / (float(iblSamples) * pdf + 0.0001);
			
			// adding a bias of 1.0 as described in GPU Gems 3 Ch20 article
			float mipLevel = roughness == 0.0 ? 0.0 : (0.5 * log2(saSample / saTexel) + 1.0); 
			
			specularLighting += textureLod(environmentMap, L, mipLevel).rgb * NdotL;
			totalWeight      += NdotL;
			
		}
	}
	specularLighting = specularLighting / totalWeight;	
	return specularLighting;
}

void main()
{             
    // retrieve data from gbuffer
    vec3 FragPos = texture(gPosition, TexCoords).rgb;
    vec3 Normal = texture(gNormal, TexCoords).rgb;
    vec4 Diffuse = texture(gDiffuse, TexCoords);
    vec4 Specular = texture(gSpecular, TexCoords);
	
	// the alpha of diffuse represent roughness of material
	float roughness = Diffuse.a;
	float metallic = Specular.a;
	// the alpha of Specular represents how "metallic" surface is.  For dia-electrics the albedo is mostly diffuse, while metals will use specular 
	vec3 albedo = mix(Diffuse.rgb, Specular.rgb, metallic);
	
	// do PBR lighting
	vec3 N = normalize(Normal);
	vec3 V = normalize(viewPos - FragPos);
	
	// calculate reflectance at normal incidence; if dia-electric (like plastic) use F0 
    // of 0.04 and if it's a metal, use the albedo color as F0 (metallic workflow)    
    vec3 F0 = vec3(0.04);
	F0 = mix(F0, albedo, metallic);
	
	// reflectance equation
    vec3 Lo = vec3(0.0);
	// calculate per-light radiance
	vec3 L = normalize(gLight.Position - FragPos);
	vec3 H = normalize(V + L);
	float distance = length(gLight.Position - FragPos);
	float attenuation = 1.0 / (distance * distance);
	vec3 radiance = gLight.Intensity * gLight.Color * attenuation;
	
	// Cook-Torrance BRDF
	float NDF = DistributionGGX(N, H, roughness);   
	float G   = GeometrySmith(N, V, L, roughness);    
	vec3 F    = FresnelSchlick(max(dot(H, V), 0.0), F0);  

	vec3 nominator    = NDF * G * F;
	float denominator = 4 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.001; // 0.001 to prevent divide by zero.
	vec3 specular = nominator / denominator;

	// kS is equal to Fresnel
	vec3 kS = F;
	// for energy conservation, the diffuse and specular light can't
	// be above 1.0 (unless the surface emits light); to preserve this
	// relationship the diffuse component (kD) should equal 1.0 - kS.
	vec3 kD = vec3(1.0) - kS;
	// multiply kD by the inverse metalness such that only non-metals 
	// have diffuse lighting, or a linear blend if partly metal (pure metals
	// have no diffuse light).
	kD *= 1.0 - metallic;	

	// scale light by NdotL
	float NdotL = max(dot(N, L), 0.0);        

	// add to outgoing radiance Lo
	// note that we already multiplied the BRDF by the Fresnel (kS) so we won't multiply by kS again
	Lo += (kD * albedo / PI + specular) * radiance * NdotL;
	
	// ambient lighting (use IBL as the ambient term)
    F = FresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);
	
	kS = F;
    kD = 1.0 - kS;
    kD *= 1.0 - metallic;
	
	vec3 irradiance = texture(irradianceMap, N).rgb;
	vec3 diffuse      = irradiance * albedo;
	
	vec3 specularIBL = SpecularIBL(N, V, roughness);
	vec2 brdf  = texture(brdfLUT, vec2(max(dot(N, V), 0.0), roughness)).rg;
    specular = specularIBL * (F * brdf.x + brdf.y);
	
	float shadowFactor = 1.0;
	if(shadowMethod == 1) {
		// calculate shadow using Moment Shadow Map
		shadowFactor = CalculateShadow4MSM(FragPos);
		// use linear step function to reduce light bleeding more
		shadowFactor = ReduceLightBleeding(shadowFactor, 0.25);	
	}
	else {
		// use standard shadow map method
		shadowFactor = CalculateShadow(FragPos, Normal);	
	}
	
	float AO = texture(ambientOcclusion, TexCoords).r;
	
	// need to saturate shadows quite a bit to make them more plausable
	shadowFactor = mix(shadowFactor, 1.0, shadowSaturation);
	vec3 ambient = (kD * diffuse + specular) * shadowFactor * AO;
	
	vec3 color = ambient + Lo;
	
	// HDR tonemapping
    color = color / (color + vec3(1.0));
    // gamma correction
    color = pow(color, vec3(1.0/2.2)); 
	
    FragColor = vec4(color , 1.0);
}