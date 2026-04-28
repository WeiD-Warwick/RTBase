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
		float etaI = iorExt, etaT = iorInt;
		float c = cosTheta;
		if (c < 0.0f) { std::swap(etaI, etaT); c = -c; }
		float eta = etaI / etaT;
		float sin2T = eta * eta * (1.0f - c * c);
		if (sin2T > 1.0f) return 1.0f;
		float cosT = sqrtf(1.0f - sin2T);
		float rs = (etaI * c - etaT * cosT) / (etaI * c + etaT * cosT);
		float rp = (etaT * c - etaI * cosT) / (etaT * c + etaI * cosT);
		return 0.5f * (rs * rs + rp * rp);
	}
	static Colour fresnelConductor(float cosTheta, Colour ior, Colour k)
	{
		cosTheta = std::min(std::max(cosTheta, 0.0f), 1.0f);
		float cos2 = cosTheta * cosTheta;
		float sin2 = 1.0f - cos2;
		auto FrChannel = [&](float eta, float kk) {
			float eta2 = eta * eta;
			float k2 = kk * kk;
			float t0 = eta2 - k2 - sin2;
			float a2b2 = sqrtf(t0 * t0 + 4.0f * eta2 * k2);
			float t1 = a2b2 + cos2;
			float a = sqrtf(0.5f * (a2b2 + t0));
			float t2 = 2.0f * cosTheta * a;
			float Rs = (t1 - t2) / (t1 + t2);

			float t3 = cos2 * a2b2 + sin2 * sin2;
			float t4 = t2 * sin2;
			float Rp = Rs * (t3 - t4) / (t3 + t4);
			return 0.5f * (Rp + Rs);
			};
		return Colour(FrChannel(ior.r, k.r), FrChannel(ior.g, k.g), FrChannel(ior.b, k.b));
	}
	static float lambdaGGX(Vec3 wi, float alpha)
	{
		float absCosTheta = fabsf(wi.z);
		float sin2Theta = std::max(1.0f - absCosTheta * absCosTheta, 0.0f);
		if (sin2Theta <= 0.0f) return 0.0f;
		float tan2Theta = sin2Theta / (absCosTheta * absCosTheta);
		float alpha2 = alpha * alpha;
		return (-1.0f + sqrtf(1.0f + alpha2 * tan2Theta)) * 0.5f;
	}
	static float Gggx(Vec3 wi, Vec3 wo, float alpha)
	{
		return 1.0f / (1.0f + lambdaGGX(wi, alpha) + lambdaGGX(wo, alpha));
	}
	static float Dggx(Vec3 h, float alpha)
	{
		if (h.z <= 0.0f) return 0.0f;
		float alpha2 = alpha * alpha;
		float cos2 = h.z * h.z;
		float tan2 = (1.0f - cos2) / cos2;
		float denom = M_PI * cos2 * cos2 * SQ(alpha2 + tan2);
		if (denom <= 0.0f) return 0.0f;
		return alpha2 / denom;
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
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec3 wiLocal = Vec3(-woLocal.x, -woLocal.y, woLocal.z);
		Vec3 wiWorld = shadingData.frame.toWorld(wiLocal);
		reflectedColour = evaluate(shadingData, wiWorld);
		pdf = 1.0f;
		return wiWorld;
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		Vec3 wrLocal = Vec3(-woLocal.x, -woLocal.y, woLocal.z);

		Vec3 delta = wiLocal - wrLocal;

		if (wiLocal.z <= 0.0f) {
			return Colour(0.0f, 0.0f, 0.0f);
		}
		if (delta.lengthSq() > EPSILON) {
			return Colour(0.0f, 0.0f, 0.0f);
		}
		return albedo->sample(shadingData.tu, shadingData.tv) / wiLocal.z;
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
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		if (woLocal.z <= 0.0f) {
			pdf = 0.0f;
			reflectedColour = Colour(0, 0, 0);
			return Vec3(0, 0, 1);
		}
		Vec3 halfwayLocal = SamplingDistributions::sampleGGXVNDF(woLocal, alpha, sampler->next(), sampler->next());
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
		Vec3 halfwayLocal = (wiLocal + woLocal).normalize();
		float D = ShadingHelper::Dggx(halfwayLocal, alpha);
		float pdfHalfway = D * halfwayLocal.z;
		float dHalfway_dWi = 1.0f / (4.0f * fabsf(Dot(wiLocal, halfwayLocal)));
		return pdfHalfway * dHalfway_dWi;
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
		float fresnelReflectance = ShadingHelper::fresnelDielectric(woLocal.z, intIOR, extIOR);
		Vec3 wiLocal;
		Vec3 wiWorld;
		if (sampler->next() < fresnelReflectance) {
			wiLocal = Vec3(-woLocal.x, -woLocal.y, woLocal.z);
			pdf = fresnelReflectance;
		}
		else {
			bool entering = woLocal.z > 0.0f;
			float etaI = entering ? extIOR : intIOR;
			float etaT = entering ? intIOR : extIOR;
			float eta = etaI / etaT;
			float cosI = fabsf(woLocal.z);
			float sin2T = eta * eta * (1.0f - cosI * cosI);
			if (sin2T >= 1.0f) {
				wiLocal = Vec3(-woLocal.x, -woLocal.y, woLocal.z);
				pdf = 1.0f;
			}
			else {
				float cosT = sqrtf(1.0f - sin2T);
				wiLocal = Vec3(-eta * woLocal.x, -eta * woLocal.y, entering ? -cosT : cosT);
				pdf = 1.0f - fresnelReflectance;
			}
		}
		wiWorld = shadingData.frame.toWorld(wiLocal);
		reflectedColour = evaluate(shadingData, wiWorld);
		return wiWorld;
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		Colour tint = albedo->sample(shadingData.tu, shadingData.tv);
		float fresnelReflectance = ShadingHelper::fresnelDielectric(woLocal.z, intIOR, extIOR);
		if (wiLocal.z * woLocal.z > 0.0f) return tint * (fresnelReflectance / fabsf(wiLocal.z));
		float etaI = woLocal.z > 0.0f ? extIOR : intIOR;
		float etaT = woLocal.z > 0.0f ? intIOR : extIOR;
		return tint * (((1.0f - fresnelReflectance) * (etaT * etaT) / (etaI * etaI)) / fabsf(wiLocal.z));
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		float F = ShadingHelper::fresnelDielectric(woLocal.z, intIOR, extIOR);
		return wiLocal.z * woLocal.z > 0.0f ? F : (1.0f - F);
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
		Vec3 halfwayLocal = SamplingDistributions::sampleGGXVNDF(woLocal, alpha, sampler->next(), sampler->next());
		float fresnelReflectance = ShadingHelper::fresnelDielectric(Dot(woLocal, halfwayLocal), intIOR, extIOR);
		Vec3 wiLocal;
		if (sampler->next() < fresnelReflectance) {
			wiLocal = (halfwayLocal * (2.0f * Dot(woLocal, halfwayLocal)) - woLocal).normalize();
		}
		else {
			bool entering = woLocal.z > 0.0f;
			float etaI = entering ? extIOR : intIOR;
			float etaT = entering ? intIOR : extIOR;
			float eta = etaI / etaT;
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
		float etaI = woLocal.z > 0.0f ? extIOR : intIOR;
		float etaT = woLocal.z > 0.0f ? intIOR : extIOR;
		Vec3 halfwayLocal = isReflection ? (wiLocal + woLocal).normalize() : (wiLocal * etaT + woLocal * etaI).normalize();
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
		float denom = SQ(etaI * dotHalfwayWo + etaT * dotHalfwayWi) * fabsf(wiLocal.z * woLocal.z);
		if (denom <= 0.0f) return Colour(0, 0, 0);
		float transmissionFactor = fabsf(dotHalfwayWi * dotHalfwayWo) * (etaT * etaT) / (etaI * etaI);
		return base * (D * G * (1.0f - fresnelReflectance) * transmissionFactor / denom);
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		if (woLocal.z == 0.0f || wiLocal.z == 0.0f) return 0.0f;
		bool isReflection = wiLocal.z * woLocal.z > 0.0f;
		float etaI = woLocal.z > 0.0f ? extIOR : intIOR;
		float etaT = woLocal.z > 0.0f ? intIOR : extIOR;
		Vec3 halfwayLocal = isReflection ? (wiLocal + woLocal).normalize() : (wiLocal * etaT + woLocal * etaI).normalize();
		if (halfwayLocal.z < 0.0f) halfwayLocal = -halfwayLocal;
		float D = ShadingHelper::Dggx(halfwayLocal, alpha);
		float pdfHalfway = D * halfwayLocal.z;
		float fresnelReflectance = ShadingHelper::fresnelDielectric(Dot(woLocal, halfwayLocal), intIOR, extIOR);
		if (isReflection) {
			float dHalfway_dWi = 1.0f / (4.0f * fabsf(Dot(wiLocal, halfwayLocal)));
			return fresnelReflectance * pdfHalfway * dHalfway_dWi;
		}
		float denom = SQ(etaI * Dot(woLocal, halfwayLocal) + etaT * Dot(wiLocal, halfwayLocal));
		float dHalfway_dWi = fabsf((etaT * etaT * Dot(wiLocal, halfwayLocal)) / denom);
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

		float cosThetaI = wiLocal.z;
		float cosThetaO = woLocal.z;

		float sinThetaI = sqrtf(1.0f - cosThetaI * cosThetaI);
		float sinThetaO = sqrtf(1.0f - cosThetaO * cosThetaO);

		float sinAlpha, tanBeta;

		if (cosThetaI > cosThetaO) {
			// theta_i < theta_o
			sinAlpha = sinThetaO;
			tanBeta = sinThetaI / cosThetaI;
		}
		else {
			sinAlpha = sinThetaI;
			tanBeta = sinThetaO / cosThetaO;
		}

		float cosPhiDiff = (wiLocal.x * woLocal.x + wiLocal.y * woLocal.y) / (sinThetaI * sinThetaO);
		cosPhiDiff = std::max(-1.0f, std::min(1.0f, cosPhiDiff));

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
			Vec3 halfwayLocal = SamplingDistributions::sampleGGXVNDF(woLocal, alpha, sampler->next(), sampler->next());
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
		return base->sample(shadingData, sampler, reflectedColour, pdf);
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		return base->evaluate(shadingData, wi);
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
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