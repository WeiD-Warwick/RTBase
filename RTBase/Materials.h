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
		// ρ / π
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
		alpha = 1.62142f * sqrtf(roughness);
	}
	bool isSmooth()
	{
		return alpha <= 0.0f;
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		if (woLocal.z <= 0.0f) {
			pdf = 0.0f;
			reflectedColour = Colour(0, 0, 0);
			return Vec3(0, 0, 1);
		}
		if (isSmooth()) {
			Vec3 wiLocal = Vec3(-woLocal.x, -woLocal.y, woLocal.z);
			Vec3 wiWorld = shadingData.frame.toWorld(wiLocal);
			reflectedColour = evaluate(shadingData, wiWorld);
			pdf = 1.0f;
			return wiWorld;
		}
		Vec3 halfwayLocal = SamplingDistributions::sampleGGXNDF(woLocal, alpha, sampler->next(), sampler->next());
		Vec3 wiLocal = (halfwayLocal * (2.0f * Dot(woLocal, halfwayLocal)) - woLocal).normalize();
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
		if (isSmooth()) {
			Vec3 wrLocal = Vec3(-woLocal.x, -woLocal.y, woLocal.z);
			Vec3 delta = wiLocal - wrLocal;
			if (delta.lengthSq() > EPSILON) return Colour(0, 0, 0);
			Colour F = ShadingHelper::fresnelConductor(fabsf(woLocal.z), eta, k);
			return albedo->sample(shadingData.tu, shadingData.tv) * F / wiLocal.z;
		}
		Vec3 halfwayLocal = (wiLocal + woLocal).normalize();
		float D = ShadingHelper::Dggx(halfwayLocal, alpha);
		float G = ShadingHelper::Gggx(wiLocal, woLocal, alpha);
		Colour F = ShadingHelper::fresnelConductor(fabsf(Dot(wiLocal, halfwayLocal)), eta, k);
		float denom = 4.0f * wiLocal.z * woLocal.z;
		if (denom <= 0.0f) return Colour(0, 0, 0);
		return albedo->sample(shadingData.tu, shadingData.tv) * F * (D * G / denom);
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		if (woLocal.z <= 0.0f || wiLocal.z <= 0.0f) return 0.0f;
		if (isSmooth()) return 0.0f;
		Vec3 halfwayLocal = (wiLocal + woLocal).normalize();
		float D = ShadingHelper::Dggx(halfwayLocal, alpha);
		float pdfHalfway = D * halfwayLocal.z;
		float dHalfway_dWi = 1.0f / (4.0f * fabsf(Dot(wiLocal, halfwayLocal)));
		return pdfHalfway * dHalfway_dWi;
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

		bool entering = woLocal.z > 0.0f;

		float etaFrom = entering ? extIOR : intIOR;
		float etaTo = entering ? intIOR : extIOR;

		float n = etaFrom / etaTo;
		float cosThetaI = fabsf(woLocal.z);
		float cosThetaISq = cosThetaI * cosThetaI;
		float sinThetaISq = 1 - cosThetaISq;
		float sinThetaI = std::sqrt(sinThetaISq);
		float sinThetaT = n * sinThetaI;

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
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		if (woLocal.z == 0.0f) {
			pdf = 0.0f;
			reflectedColour = Colour(0, 0, 0);
			return Vec3(0, 0, 1);
		}
		Vec3 halfwayLocal = SamplingDistributions::sampleGGXNDF(woLocal, alpha, sampler->next(), sampler->next());
		float fresnelReflectance = ShadingHelper::fresnelDielectric(Dot(woLocal, halfwayLocal), intIOR, extIOR);
		Vec3 wiLocal;
		if (sampler->next() < fresnelReflectance) {
			wiLocal = (halfwayLocal * (2.0f * Dot(woLocal, halfwayLocal)) - woLocal).normalize();
		}
		else {
			bool entering = woLocal.z > 0.0f;
			float fromIOR = entering ? extIOR : intIOR;
			float toIOR = entering ? intIOR : extIOR;
			float eta = fromIOR / toIOR;
			float cosOH = Dot(woLocal, halfwayLocal);
			float sin2T = eta * eta * (1.0f - cosOH * cosOH);
			if (sin2T >= 1.0f) wiLocal = (halfwayLocal * (2.0f * cosOH) - woLocal).normalize();
			else {
				float cosT = sqrtf(1.0f - sin2T);
				float sign = cosOH > 0.0f ? 1.0f : -1.0f;
				wiLocal = (halfwayLocal * (eta * cosOH - sign * cosT) - woLocal * eta).normalize();
			}
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
		if (woLocal.z == 0.0f || wiLocal.z == 0.0f) return Colour(0, 0, 0);

		bool isReflection = wiLocal.z * woLocal.z > 0.0f;
		float fromIOR = woLocal.z > 0.0f ? extIOR : intIOR;
		float toIOR = woLocal.z > 0.0f ? intIOR : extIOR;
		Vec3 halfwayLocal = isReflection ? (wiLocal + woLocal).normalize() : (wiLocal * toIOR + woLocal * fromIOR).normalize();
		if (halfwayLocal.z < 0.0f) halfwayLocal = -halfwayLocal;

		float D = ShadingHelper::Dggx(halfwayLocal, alpha);
		float G = ShadingHelper::Gggx(wiLocal, woLocal, alpha);
		float fresnelReflectance = ShadingHelper::fresnelDielectric(Dot(woLocal, halfwayLocal), intIOR, extIOR);
		Colour base = albedo->sample(shadingData.tu, shadingData.tv);
		if (isReflection) {
			float denom = 4.0f * fabsf(wiLocal.z * woLocal.z);
			if (denom <= 0.0f) return Colour(0, 0, 0);
			return base * (D * G * fresnelReflectance / denom);
		}
		float dotHalfwayWi = Dot(wiLocal, halfwayLocal);
		float dotHalfwayWo = Dot(woLocal, halfwayLocal);
		float denom = SQ(fromIOR * dotHalfwayWo + toIOR * dotHalfwayWi) * fabsf(wiLocal.z * woLocal.z);
		if (denom <= 0.0f) return Colour(0, 0, 0);
		float transmissionFactor = fabsf(dotHalfwayWi * dotHalfwayWo) * (toIOR * toIOR) / (fromIOR * fromIOR);
		return base * (D * G * (1.0f - fresnelReflectance) * transmissionFactor / denom);
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		if (woLocal.z == 0.0f || wiLocal.z == 0.0f) return 0.0f;
		bool isReflection = wiLocal.z * woLocal.z > 0.0f;
		float fromIOR = woLocal.z > 0.0f ? extIOR : intIOR;
		float toIOR = woLocal.z > 0.0f ? intIOR : extIOR;
		Vec3 halfwayLocal = isReflection ? (wiLocal + woLocal).normalize() : (wiLocal * toIOR + woLocal * fromIOR).normalize();
		if (halfwayLocal.z < 0.0f) halfwayLocal = -halfwayLocal;
		float D = ShadingHelper::Dggx(halfwayLocal, alpha);
		float pdfHalfway = D * halfwayLocal.z;
		float fresnelReflectance = ShadingHelper::fresnelDielectric(Dot(woLocal, halfwayLocal), intIOR, extIOR);
		if (isReflection) {
			float dHalfway_dWi = 1.0f / (4.0f * fabsf(Dot(wiLocal, halfwayLocal)));
			return fresnelReflectance * pdfHalfway * dHalfway_dWi;
		}
		float denom = SQ(fromIOR * Dot(woLocal, halfwayLocal) + toIOR * Dot(wiLocal, halfwayLocal));
		float dHalfway_dWi = fabsf((toIOR * toIOR * Dot(wiLocal, halfwayLocal)) / denom);
		return (1.0f - fresnelReflectance) * pdfHalfway * dHalfway_dWi;
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

		Colour lambert = albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
		float sigmaSq = sigma * sigma;

		float A = 1.0f - sigmaSq / (2.0f * (sigmaSq + 0.33f));
		float B = 0.45f * sigmaSq / (sigmaSq + 0.09f);

		float cosThfromIOR = wiLocal.z;
		float cosThetaO = woLocal.z;

		float sinThfromIOR = sqrtf(std::max(0.0f, 1.0f - cosThfromIOR * cosThfromIOR));
		float sinThetaO = sqrtf(std::max(0.0f, 1.0f - cosThetaO * cosThetaO));

		float sinAlpha, tanBeta;

		if (cosThfromIOR > cosThetaO) {
			// theta_i < theta_o
			sinAlpha = sinThetaO;
			tanBeta = sinThfromIOR / cosThfromIOR;
		}
		else {
			sinAlpha = sinThfromIOR;
			tanBeta = sinThetaO / cosThetaO;
		}

		float sinThetaProduct = sinThfromIOR * sinThetaO;
		float cosPhiDiff = 0.0f;
		if (sinThetaProduct > EPSILON) {
			cosPhiDiff = (wiLocal.x * woLocal.x + wiLocal.y * woLocal.y) / sinThetaProduct;
			cosPhiDiff = std::max(-1.0f, std::min(1.0f, cosPhiDiff));
		}

		float orenNayar = A + B * std::max(0.0f, cosPhiDiff) * sinAlpha * tanBeta;

		return lambert * orenNayar;
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
		float fresnelReflectance = ShadingHelper::fresnelDielectric(fabsf(woLocal.z), intIOR, extIOR);
		float specularProbability = std::min(std::max(fresnelReflectance, 0.05f), 0.95f);
		Vec3 wiLocal;
		if (sampler->next() < specularProbability) {
			Vec3 halfwayLocal = SamplingDistributions::sampleGGXNDF(woLocal, alpha, sampler->next(), sampler->next());
			wiLocal = (halfwayLocal * (2.0f * Dot(woLocal, halfwayLocal)) - woLocal).normalize();
		}
		else {
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
		float fresnelReflectance = ShadingHelper::fresnelDielectric(fabsf(woLocal.z), intIOR, extIOR);
		Colour diffuseAlbedo = albedo->sample(shadingData.tu, shadingData.tv) * (1.0f - fresnelReflectance);
		Colour diffuse = diffuseAlbedo / M_PI;
		Vec3 halfwayLocal = (wiLocal + woLocal).normalize();
		float D = ShadingHelper::Dggx(halfwayLocal, alpha);
		float G = ShadingHelper::Gggx(wiLocal, woLocal, alpha);
		float specularFresnel = ShadingHelper::fresnelDielectric(fabsf(Dot(wiLocal, halfwayLocal)), intIOR, extIOR);
		float denom = 4.0f * wiLocal.z * woLocal.z;
		Colour specular = denom > 0.0f ? Colour(1, 1, 1) * (D * G * specularFresnel / denom) : Colour(0, 0, 0);
		return diffuse + specular;
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		if (woLocal.z <= 0.0f || wiLocal.z <= 0.0f) return 0.0f;
		float fresnelReflectance = ShadingHelper::fresnelDielectric(fabsf(woLocal.z), intIOR, extIOR);
		float specularProbability = std::min(std::max(fresnelReflectance, 0.05f), 0.95f);
		float diffusePdf = SamplingDistributions::cosineHemispherePDF(wiLocal);
		Vec3 halfwayLocal = (wiLocal + woLocal).normalize();
		float D = ShadingHelper::Dggx(halfwayLocal, alpha);
		float specularPdf = (D * halfwayLocal.z) / (4.0f * fabsf(Dot(wiLocal, halfwayLocal)));
		return specularProbability * specularPdf + (1.0f - specularProbability) * diffusePdf;
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
