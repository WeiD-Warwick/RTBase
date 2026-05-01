#pragma once

#include "Core.h"
#include "Imaging.h"
#include "Sampling.h"

#pragma warning( disable : 4244)
#pragma warning( disable : 4305) // Double to float

class BSDF;

class ShadingData
{
public:
	Vec3 x;
	Vec3 wo;
	Vec3 sNormal;
	Vec3 gNormal;
	float tu;
	float tv;
	Frame frame;
	BSDF* bsdf;
	float t;
	ShadingData() {}
	ShadingData(Vec3 _x, Vec3 n)
	{
		x = _x;
		gNormal = n;
		sNormal = n;
		bsdf = NULL;
	}
};

class ShadingHelper
{
public:
	static float fresnelDielectric(float cosTheta, float iorInt, float iorExt) {
		cosTheta = std::min(std::max(cosTheta, -1.0f), 1.0f);
		float n = iorExt / iorInt;
		// n depends on direction
		if (cosTheta < 0.0f) {
			n = 1.0f / n;
			cosTheta = -cosTheta;
		}

		// Rewrite Snell's Law
		float sinTheta = sqrtf(std::max(0.0f, 1.0f - cosTheta * cosTheta));
		float sinThtoIOR = n * sinTheta;

		if (sinThtoIOR >= 1.0f) return 1.0f;

		// Calculate thtoIOR
		float cosThtoIOR = sqrtf(1.0f - sinThtoIOR * sinThtoIOR);

		// Parallel
		float parallel = (cosTheta - n * cosThtoIOR) / (cosTheta + n * cosThtoIOR);
		// Perpendicular
		float perpendicular = (n * cosTheta - cosThtoIOR) / (n * cosTheta + cosThtoIOR);

		// Average
		return (parallel * parallel + perpendicular * perpendicular) * 0.5f;
	}

	static Colour fresnelConductor(float cosTheta, Colour ior, Colour k) {
		float cosThetaSq = cosTheta * cosTheta;
		float sinThetaSq = 1.0f - cosThetaSq;

		Colour a = ior * ior + k * k;
		Colour b = ior * 2.0f * cosTheta;
		Colour sinSq = Colour(sinThetaSq, sinThetaSq, sinThetaSq);
		Colour cosSq = Colour(cosThetaSq, cosThetaSq, cosThetaSq);

		// parallel
		Colour parallel = (a * cosThetaSq - b + sinSq) / (a * cosThetaSq + b + sinSq);
		// perpendicular
		Colour perpendicular = (a - b + cosSq) / (a + b + cosSq);
		// Average
		return (parallel + perpendicular) * 0.5f;
	}
	static float lambdaGGX(Vec3 wi, float alpha)
	{
		if (wi.z <= 0.0f) return 0.0f;
		float cosThetaSq = wi.z * wi.z;
		float sinThetaSq = std::max(0.0f, 1.0f - cosThetaSq);
		float tanThetaSq = sinThetaSq / cosThetaSq;
		float alphaSq = alpha * alpha;
		return (sqrtf(1.0f + alphaSq * tanThetaSq) - 1.0f) * 0.5f;
	}
	static float Gggx(Vec3 wi, Vec3 wo, float alpha)
	{
		float wiG = 1.0f / (1.0f + lambdaGGX(wi, alpha));
		float woG = 1.0f / (1.0f + lambdaGGX(wo, alpha));
		return wiG * woG;
	}
	// GGX distribution
	static float Dggx(Vec3 h, float alpha)
	{
		if (h.z <= 0.0f) return 0.0f;
		float alphaSq = alpha * alpha;
		float cosThetaSq = h.z * h.z;
		float denom = cosThetaSq * (alphaSq - 1.0f) + 1.0f;
		denom = M_PI * denom * denom;
		if (denom <= 0.0f) return 0.0f;
		return alphaSq / denom;
	}
};

class BSDF
{
public:
	Colour emission;
	virtual Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf) = 0;
	virtual Colour evaluate(const ShadingData& shadingData, const Vec3& wi) = 0;
	virtual float PDF(const ShadingData& shadingData, const Vec3& wi) = 0;
	virtual bool isPureSpecular() = 0;
	virtual bool isTwoSided() = 0;
	bool isLight()
	{
		return emission.Lum() > 0 ? true : false;
	}
	void addLight(Colour _emission)
	{
		emission = _emission;
	}
	Colour emit(const ShadingData& shadingData, const Vec3& wi)
	{
		return emission;
	}
	virtual float mask(const ShadingData& shadingData) = 0;
};


class DiffuseBSDF : public BSDF
{
public:
	Texture* albedo;
	DiffuseBSDF() = default;
	DiffuseBSDF(Texture* _albedo)
	{
		albedo = _albedo;
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		Vec3 wiLocal = SamplingDistributions::cosineSampleHemisphere(sampler->next(), sampler->next());
		pdf = SamplingDistributions::cosineHemispherePDF(wiLocal);
		Vec3 wiWorld = shadingData.frame.toWorld(wiLocal);
		reflectedColour = evaluate(shadingData, wiWorld);
		return wiWorld;
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		Vec3 wiLocal = shadingData.frame.toLocal(wi);

		if (wiLocal.z <= 0.0f) {
			return Colour(0.0f, 0.0f, 0.0f);
		}
		// Lambert BRDF
		return albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		Vec3 wiLocal = shadingData.frame.toLocal(wi);

		if (wiLocal.z <= 0.0f) {
			return 0.0f;
		}
		return SamplingDistributions::cosineHemispherePDF(wiLocal);
	}
	bool isPureSpecular()
	{
		return false;
	}
	bool isTwoSided()
	{
		return true;
	}
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};

class MirrorBSDF : public BSDF
{
public:
	Texture* albedo;
	MirrorBSDF() = default;
	MirrorBSDF(Texture* _albedo)
	{
		albedo = _albedo;
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		// Convert shadingData.wo to local space 
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		// Perfect mirror reflection in local space
		Vec3 wiLocal = reflectLocal(woLocal);
		// Convert back to world space
		Vec3 wiWorld = shadingData.frame.toWorld(wiLocal);
		// Get reflectedColour
		reflectedColour = evaluateMirror(shadingData, wiLocal);
		pdf = 1.0f;
		return wiWorld;
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		Vec3 wrLocal = reflectLocal(woLocal);

		// Dirac delta
		if (!isMirrorDirection(wiLocal, wrLocal)) {
			return Colour(0.0f, 0.0f, 0.0f);
		}
		return evaluateMirror(shadingData, wiLocal);
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		return 0.0f;
	}
	bool isPureSpecular()
	{
		return true;
	}
	bool isTwoSided()
	{
		return true;
	}
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
private:
	Vec3 reflectLocal(const Vec3& woLocal) {
		// Reflect x and y
		// wr = (-wx, -wy, wz)
		return Vec3(-woLocal.x, -woLocal.y, woLocal.z);
	}

	bool isMirrorDirection(const Vec3& wiLocal, const Vec3& wrLocal) {
		Vec3 delta = wiLocal - wrLocal;
		return delta.lengthSq() <= EPSILON;
	}

	Colour evaluateMirror(const ShadingData& shadingData, const Vec3& wiLocal) {
		// wr · n = wiLocal.z
		float cosTheta = wiLocal.z;

		if (cosTheta <= EPSILON) {
			return Colour(0.0f, 0.0f, 0.0f);
		}

		Colour reflectedColour = albedo->sample(shadingData.tu, shadingData.tv);

		// Perfect mirror
		return reflectedColour / cosTheta;
	}
};

class ConductorBSDF : public BSDF
{
public:
	Texture* albedo;
	Colour eta;
	Colour k;
	float alpha;
	ConductorBSDF() = default;
	ConductorBSDF(Texture* _albedo, Colour _eta, Colour _k, float roughness)
	{
		albedo = _albedo;
		eta = _eta;
		k = _k;
		alpha = roughness < EPSILON ? 0.0f : 1.62142f * sqrtf(roughness);
	}
	bool isSmooth()
	{
		return alpha <= EPSILON;
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		if (woLocal.z <= 0.0f) {
			pdf = 0.0f;
			reflectedColour = Colour(0, 0, 0);
			return Vec3(0, 0, 1);
		}
		// Smooth conductor
		if (isSmooth()) {
			Vec3 wrLocal(-woLocal.x, -woLocal.y, woLocal.z);
			Vec3 wrWorld = shadingData.frame.toWorld(wrLocal);
			reflectedColour = evaluate(shadingData, wrWorld);
			pdf = 1.0f;
			return wrWorld;
		}

		// GGX half-vector sampling
		Vec3 wmLocal = SamplingDistributions::sampleGGX(woLocal, alpha, sampler->next(), sampler->next());
		Vec3 wiLocal = (-woLocal + wmLocal * (2.0f * Dot(woLocal, wmLocal))).normalize();

		if (wiLocal.z <= 0.0f) {
			pdf = 0.0f;
			reflectedColour = Colour(0, 0, 0);
			return shadingData.frame.toWorld(wiLocal);
		}

		Vec3 wiWorld = shadingData.frame.toWorld(wiLocal);
		reflectedColour = evaluate(shadingData, wiWorld);
		pdf = PDF(shadingData, wiWorld);
		return wiWorld;
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		if (woLocal.z <= 0.0f || wiLocal.z <= 0.0f) return Colour(0, 0, 0);

		Colour Le = albedo->sample(shadingData.tu, shadingData.tv);
		if (isSmooth()) {
			Vec3 wrLocal = Vec3(-woLocal.x, -woLocal.y, woLocal.z);
			Vec3 deltaLocal = wiLocal - wrLocal;
			if (deltaLocal.lengthSq() > EPSILON) return Colour(0, 0, 0);
			Colour F = ShadingHelper::fresnelConductor(fabsf(woLocal.z), eta, k);
			return Le * F / wiLocal.z;
		}

		// Cook-Torrance
		Vec3 wmLocal = (wiLocal + woLocal).normalize();
		float D = ShadingHelper::Dggx(wmLocal, alpha);
		float G = ShadingHelper::Gggx(wiLocal, woLocal, alpha);
		Colour F = ShadingHelper::fresnelConductor(fabsf(Dot(woLocal, wmLocal)), eta, k);
		float denominator = 4.0f * wiLocal.z * woLocal.z;
		if (denominator <= 0.0f) return Colour(0, 0, 0);
		return Le * F * (D * G / denominator);
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		if (woLocal.z <= 0.0f || wiLocal.z <= 0.0f) return 0.0f;
		if (isSmooth()) return 0.0f;
		// Half-vector PDF transform
		Vec3 wmLocal = (wiLocal + woLocal).normalize();
		float D = ShadingHelper::Dggx(wmLocal, alpha);
		float pdfWm = D * wmLocal.z;
		float dWmDWo = 1.0f / (4.0f * Dot(woLocal, wmLocal));
		return fabsf(pdfWm * dWmDWo);
	}
	bool isPureSpecular()
	{
		return isSmooth();
	}
	bool isTwoSided()
	{
		return true;
	}
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};

class GlassBSDF : public BSDF
{
public:
	Texture* albedo;
	float intIOR;
	float extIOR;
	GlassBSDF() = default;
	GlassBSDF(Texture* _albedo, float _intIOR, float _extIOR)
	{
		albedo = _albedo;
		intIOR = _intIOR;
		extIOR = _extIOR;
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Colour Le = albedo->sample(shadingData.tu, shadingData.tv);

		// judge entering or exiting
		bool entering = woLocal.z > 0.0f;

		float etaFrom = entering ? extIOR : intIOR;
		float etaTo = entering ? intIOR : extIOR;

		float n = etaFrom / etaTo;
		float cosThetaI = fabsf(woLocal.z);
		float cosThetaISq = cosThetaI * cosThetaI;
		float sinThetaISq = 1 - cosThetaISq;
		float sinThetaI = std::sqrt(sinThetaISq);
		float sinThetaT = n * sinThetaI;

		// All internal reflection
		if (sinThetaT >= 1.0f) {
			pdf = 1.0f;
			reflectedColour = Le / cosThetaI;
			Vec3 wrLocal(-woLocal.x, -woLocal.y, woLocal.z);
			return shadingData.frame.toWorld(wrLocal);
		}

		float F = ShadingHelper::fresnelDielectric(woLocal.z, intIOR, extIOR);
		float r = sampler->next();
		if (r < F) {
			// Reflection
			pdf = F;
			reflectedColour = Le * F / cosThetaI;
			Vec3 wrLocal(-woLocal.x, -woLocal.y, woLocal.z);
			return shadingData.frame.toWorld(wrLocal);
		}

		// Transmission
		pdf = 1.0f - F;

		float sinThetaTSq = sinThetaT * sinThetaT;
		float cosThetaT = std::sqrt(1.0f - sinThetaTSq);
		float zSign = woLocal.z > 0.0f ? -1.0f : 1.0f;
		float scale = (extIOR / intIOR) * (extIOR * intIOR);
		reflectedColour = Le * (1.0f - F) * scale / cosThetaT;
		Vec3 wtLocal = Vec3(-n * woLocal.x, -n * woLocal.y, zSign * cosThetaT);

		return shadingData.frame.toWorld(wtLocal);
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		return Colour(0.f, 0.f, 0.f);
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		return 0.0f;
	}
	bool isPureSpecular()
	{
		return true;
	}
	bool isTwoSided()
	{
		return false;
	}
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};

class DielectricBSDF : public BSDF
{
public:
	Texture* albedo;
	float intIOR;
	float extIOR;
	float alpha;
	DielectricBSDF() = default;
	DielectricBSDF(Texture* _albedo, float _intIOR, float _extIOR, float roughness)
	{
		albedo = _albedo;
		intIOR = _intIOR;
		extIOR = _extIOR;
		alpha = 1.62142f * sqrtf(roughness);
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		Vec3 wi = SamplingDistributions::cosineSampleHemisphere(sampler->next(), sampler->next());
		pdf = wi.z / M_PI;
		reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
		wi = shadingData.frame.toWorld(wi);
		return wi;
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with Dielectric evaluation code
		return albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with Dielectric PDF
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		return SamplingDistributions::cosineHemispherePDF(wiLocal);
	}
	bool isPureSpecular()
	{
		return false;
	}
	bool isTwoSided()
	{
		return false;
	}
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};

class OrenNayarBSDF : public BSDF
{
public:
	Texture* albedo;
	float sigma;
	OrenNayarBSDF() = default;
	OrenNayarBSDF(Texture* _albedo, float _sigma)
	{
		albedo = _albedo;
		sigma = _sigma;
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		Vec3 wiLocal = SamplingDistributions::cosineSampleHemisphere(sampler->next(), sampler->next());
		Vec3 wiWorld = shadingData.frame.toWorld(wiLocal);
		reflectedColour = evaluate(shadingData, wiWorld);
		pdf = SamplingDistributions::cosineHemispherePDF(wiLocal);
		return wiWorld;
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);

		if (wiLocal.z <= 0.0f || woLocal.z <= 0.0f) {
			return Colour(0.0f, 0.0f, 0.0f);
		}

		// Oren-Nayar diffuse base
		Colour Le = albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
		float sigmaSq = sigma * sigma;

		float A = 1.0f - sigmaSq / (2.0f * (sigmaSq + 0.33f));
		float B = 0.45f * sigmaSq / (sigmaSq + 0.09f);

		float cosThetaI = wiLocal.z;
		float cosThetaO = woLocal.z;

		float sinThetaI = sqrtf(1.0f - cosThetaI * cosThetaI);
		float sinThetaO = sqrtf(1.0f - cosThetaO * cosThetaO);

		float sinAlpha, tanBeta;

		// View and light angle split
		if (cosThetaI > cosThetaO) {
			// theta_i < theta_o
			sinAlpha = sinThetaO;
			tanBeta = sinThetaI / cosThetaI;
		}
		else {
			sinAlpha = sinThetaI;
			tanBeta = sinThetaO / cosThetaO;
		}

		float sinThetaProduct = sinThetaI * sinThetaO;
		float cosPhiDiff = 0.0f;
		if (sinThetaProduct > EPSILON) {
			cosPhiDiff = (wiLocal.x * woLocal.x + wiLocal.y * woLocal.y) / sinThetaProduct;
		}

		float orenNayar = A + B * std::max(0.0f, cosPhiDiff) * sinAlpha * tanBeta;

		return Le * orenNayar;
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		if (wiLocal.z <= 0.0f) {
			return 0.0f;
		}
		return SamplingDistributions::cosineHemispherePDF(wiLocal);
	}
	bool isPureSpecular()
	{
		return false;
	}
	bool isTwoSided()
	{
		return true;
	}
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};

// Phong
class PlasticBSDF : public BSDF
{
public:
	Texture* albedo;
	float intIOR;
	float extIOR;
	float alpha;
	PlasticBSDF() = default;
	PlasticBSDF(Texture* _albedo, float _intIOR, float _extIOR, float roughness)
	{
		albedo = _albedo;
		intIOR = _intIOR;
		extIOR = _extIOR;
		alpha = 1.62142f * sqrtf(roughness);
	}
	float alphaToPhongExponent()
	{
		return (2.0f / SQ(std::max(alpha, 0.001f))) - 2.0f;
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		// Fresnel sample weight
		float F = ShadingHelper::fresnelDielectric(fabsf(woLocal.z), intIOR, extIOR);
		float r = sampler->next();
		Vec3 wiLocal;

		if (r < F) {
			// Phong lobe sampling
			float phongExponent = alphaToPhongExponent();
			float cosTheta = powf(sampler->next(), 1.0f / (phongExponent + 1.0f));
			float sinTheta = sqrtf(std::max(0.0f, 1.0f - cosTheta * cosTheta));
			float phi = 2.0f * M_PI * sampler->next();
			Vec3 wiLobe(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);
			Vec3 wrLocal(-woLocal.x, -woLocal.y, woLocal.z);
			Frame wrFrame;
			wrFrame.fromVector(wrLocal);
			wiLocal = wrFrame.toWorld(wiLobe);
		} else {
			// Diffuse sampling
			wiLocal = SamplingDistributions::cosineSampleHemisphere(sampler->next(), sampler->next());
		}

		Vec3 wiWorld = shadingData.frame.toWorld(wiLocal);
		reflectedColour = evaluate(shadingData, wiWorld);
		pdf = PDF(shadingData, wiWorld);
		return wiWorld;
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		if (woLocal.z <= 0.0f || wiLocal.z <= 0.0f) return Colour(0, 0, 0);

		// mix Diffuse and Glossy
		float F = ShadingHelper::fresnelDielectric(fabsf(woLocal.z), intIOR, extIOR);
		float phongExponent = alphaToPhongExponent();
		Vec3 wrLocal(-woLocal.x, -woLocal.y, woLocal.z);
		float wrDotWi = std::max(0.0f, Dot(wrLocal, wiLocal));
		Colour diffuse = albedo->sample(shadingData.tu, shadingData.tv) * (1.0f - F) / M_PI;
		Colour glossy = Colour(1, 1, 1) * F * ((phongExponent + 2.0f) / (2.0f * M_PI)) * powf(wrDotWi, phongExponent);
		return diffuse + glossy;
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		if (woLocal.z <= 0.0f || wiLocal.z <= 0.0f) return 0.0f;

		// Mixture PDF
		float F = ShadingHelper::fresnelDielectric(fabsf(woLocal.z), intIOR, extIOR);
		float phongExponent = alphaToPhongExponent();
		Vec3 wrLocal(-woLocal.x, -woLocal.y, woLocal.z);
		float wrDotWi = std::max(0.0f, Dot(wrLocal, wiLocal));
		float diffusePdf = SamplingDistributions::cosineHemispherePDF(wiLocal);
		float glossyPdf = ((phongExponent + 1.0f) / (2.0f * M_PI)) * powf(wrDotWi, phongExponent);
		return F * glossyPdf + (1.0f - F) * diffusePdf;
	}
	bool isPureSpecular()
	{
		return false;
	}
	bool isTwoSided()
	{
		return true;
	}
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};

class LayeredBSDF : public BSDF
{
public:
	BSDF* base;
	Colour sigmaa;
	float thickness;
	float intIOR;
	float extIOR;
	LayeredBSDF() = default;
	LayeredBSDF(BSDF* _base, Colour _sigmaa, float _thickness, float _intIOR, float _extIOR)
	{
		base = _base;
		sigmaa = _sigmaa;
		thickness = _thickness;
		intIOR = _intIOR;
		extIOR = _extIOR;
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		// Add code to include layered sampling
		return base->sample(shadingData, sampler, reflectedColour, pdf);
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		// Add code for evaluation of layer
		return base->evaluate(shadingData, wi);
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		// Add code to include PDF for sampling layered BSDF
		return base->PDF(shadingData, wi);
	}
	bool isPureSpecular()
	{
		return base->isPureSpecular();
	}
	bool isTwoSided()
	{
		return true;
	}
	float mask(const ShadingData& shadingData)
	{
		return base->mask(shadingData);
	}
};
