#pragma once

#include "Core.h"
#include "Geometry.h"
#include "Materials.h"
#include "Sampling.h"

#pragma warning( disable : 4244)

class SceneBounds
{
public:
	Vec3 sceneCentre;
	float sceneRadius;
};

class Light
{
public:
	virtual Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& emittedColour, float& pdf) = 0;
	virtual Colour evaluate(const Vec3& wi) = 0;
	virtual float PDF(const ShadingData& shadingData, const Vec3& wi) = 0;
	virtual bool isArea() = 0;
	virtual Vec3 normal(const ShadingData& shadingData, const Vec3& wi) = 0;
	virtual float totalIntegratedPower() = 0;
	virtual Vec3 samplePositionFromLight(Sampler* sampler, float& pdf) = 0;
	virtual Vec3 sampleDirectionFromLight(Sampler* sampler, float& pdf) = 0;
};

class AreaLight : public Light
{
public:
	Triangle* triangle = NULL;
	Colour emission;
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& emittedColour, float& pdf)
	{
		emittedColour = emission;
		return triangle->sample(sampler, pdf);
	}
	Colour evaluate(const Vec3& wi)
	{
		if (Dot(wi, triangle->gNormal()) < 0)
		{
			return emission;
		}
		return Colour(0.0f, 0.0f, 0.0f);
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		return 1.0f / triangle->area;
	}
	bool isArea()
	{
		return true;
	}
	Vec3 normal(const ShadingData& shadingData, const Vec3& wi)
	{
		return triangle->gNormal();
	}
	float totalIntegratedPower()
	{
		return (triangle->area * emission.Lum());
	}
	Vec3 samplePositionFromLight(Sampler* sampler, float& pdf)
	{
		return triangle->sample(sampler, pdf);
	}
	Vec3 sampleDirectionFromLight(Sampler* sampler, float& pdf)
	{
		// Add code to sample a direction from the light
		Vec3 wi = Vec3(0, 0, 1);
		pdf = 1.0f;
		Frame frame;
		frame.fromVector(triangle->gNormal());
		return frame.toWorld(wi);
	}
};

class BackgroundColour : public Light
{
public:
	Colour emission;
	BackgroundColour(Colour _emission)
	{
		emission = _emission;
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		Vec3 wi = SamplingDistributions::uniformSampleSphere(sampler->next(), sampler->next());
		pdf = SamplingDistributions::uniformSpherePDF(wi);
		reflectedColour = emission;
		return wi;
	}
	Colour evaluate(const Vec3& wi)
	{
		return emission;
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		return SamplingDistributions::uniformSpherePDF(wi);
	}
	bool isArea()
	{
		return false;
	}
	Vec3 normal(const ShadingData& shadingData, const Vec3& wi)
	{
		return -wi;
	}
	float totalIntegratedPower()
	{
		return emission.Lum() * 4.0f * M_PI;
	}
	Vec3 samplePositionFromLight(Sampler* sampler, float& pdf)
	{
		Vec3 p = SamplingDistributions::uniformSampleSphere(sampler->next(), sampler->next());
		p = p * use<SceneBounds>().sceneRadius;
		p = p + use<SceneBounds>().sceneCentre;
		pdf = 4 * M_PI * use<SceneBounds>().sceneRadius * use<SceneBounds>().sceneRadius;
		return p;
	}
	Vec3 sampleDirectionFromLight(Sampler* sampler, float& pdf)
	{
		Vec3 wi = SamplingDistributions::uniformSampleSphere(sampler->next(), sampler->next());
		pdf = SamplingDistributions::uniformSpherePDF(wi);
		return wi;
	}
};

class EnvironmentMap : public Light
{
public:
	Texture* env;
	// [row1, row2, row3, ...]
	std::vector<float> cdf;
	float weightSum;

	EnvironmentMap(Texture* _env) {
		env = _env;
		buildCDF();
	}

	void buildCDF() {
		int width = env->width;
		int height = env->height;

		cdf.resize(width * height);
		weightSum = 0.0f;

		for (int y = 0; y < height; y++) {
			float v = (y + 0.5f) / height;
			float theta = M_PI * v;
			float sinTheta = sinf(theta);

			for (int x = 0; x < width; x++) {
				int idx = y * width + x;

				float lum = env->texels[idx].Lum();
				float weight = lum * sinTheta;

				weightSum += weight;
				cdf[idx] = weightSum;
			}
		}
	}

	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{

		// Sample a pixel from CDFs
		int width = env->width;
		int height = env->height;
		float xi = sampler->next() * weightSum;
		int idx = std::upper_bound(cdf.begin(), cdf.end(), xi) - cdf.begin();

		if (idx >= width * height) {
			idx = width * height - 1;
		}

		// Convert idx to xy
		int py = idx / width;
		int px = idx - py * width;

		// Assignment: Update this code to importance sampling lighting based on luminance of each pixel
		float u = (px + sampler->next()) / width;
		float v = (py + sampler->next()) / height;

		float theta = M_PI * v;
		float phi = 2.0f * M_PI * u;

		float sinTheta;
		Vec3 wi = uvToDirection(u, v, sinTheta);

		pdf = pdfFromPixel(idx, sinTheta);

		if (pdf <= 0.0f) {
			reflectedColour = Colour(0.0f, 0.0f, 0.0f);
			return wi;
		}

		reflectedColour = evaluate(wi);

		return wi;
	}
	Colour evaluate(const Vec3& wi)
	{
		float u = atan2f(wi.z, wi.x);
		u = (u < 0.0f) ? u + (2.0f * M_PI) : u;
		u = u / (2.0f * M_PI);
		float v = acosf(wi.y) / M_PI;
		return env->sample(u, v);
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		// Assignment: Update this code to return the correct PDF of luminance weighted importance sampling
		float sinTheta;
		int idx = directionToPixel(wi, sinTheta);

		return pdfFromPixel(idx, sinTheta);
	}
	bool isArea()
	{
		return false;
	}
	Vec3 normal(const ShadingData& shadingData, const Vec3& wi)
	{
		return -wi;
	}
	float totalIntegratedPower()
	{
		float total = 0;
		for (int i = 0; i < env->height; i++)
		{
			float st = sinf(((float)i / (float)env->height) * M_PI);
			for (int n = 0; n < env->width; n++)
			{
				total += (env->texels[(i * env->width) + n].Lum() * st);
			}
		}
		total = total / (float)(env->width * env->height);
		return total * 4.0f * M_PI;
	}
	Vec3 samplePositionFromLight(Sampler* sampler, float& pdf)
	{
		// Samples a point on the bounding sphere of the scene. Feel free to improve this.
		Vec3 p = SamplingDistributions::uniformSampleSphere(sampler->next(), sampler->next());
		p = p * use<SceneBounds>().sceneRadius;
		p = p + use<SceneBounds>().sceneCentre;
		pdf = 1.0f / (4 * M_PI * SQ(use<SceneBounds>().sceneRadius));
		return p;
	}
	Vec3 sampleDirectionFromLight(Sampler* sampler, float& pdf)
	{
		// Replace this tabulated sampling of environment maps
		Vec3 wi = SamplingDistributions::uniformSampleSphere(sampler->next(), sampler->next());
		pdf = SamplingDistributions::uniformSpherePDF(wi);
		return wi;
	}
	Vec3 uvToDirection(float u, float v, float& sinTheta) {
		float theta = M_PI * v;
		float phi = 2.0f * M_PI * u;

		sinTheta = sinf(theta);
		float cosTheta = cosf(theta);

		Vec3 wi;
		wi.x = sinTheta * cosf(phi);
		wi.y = cosTheta;
		wi.z = sinTheta * sinf(phi);

		return wi.normalize();
	}

	int directionToPixel(const Vec3& wi, float& sinTheta) {
		int width = env->width;
		int height = env->height;

		Vec3 w = wi.normalize();

		float phi = atan2f(w.z, w.x);
		if (phi < 0.0f) {
			phi += 2.0f * M_PI;
		}
		float u = phi / (2.0f * M_PI);
		float y = std::max(-1.0f, std::min(1.0f, w.y));
		float theta = acosf(y);
		float v = theta / M_PI;
		sinTheta = sinf(theta);

		int px = u * width;
		int py = v * height;

		if (px < 0) px = 0;
		if (px >= width) px = width - 1;

		if (py < 0) py = 0;
		if (py >= height) py = height - 1;

		return py * width + px;
	}

	float pdfFromPixel(int idx, float sinTheta) {
		int width = env->width;
		int height = env->height;

		if (weightSum <= EPSILON || cdf.empty()) {
			return 0.0f;
		}

		if (idx < 0 || idx >= int(cdf.size())) {
			return 0.0f;
		}

		if (sinTheta <= EPSILON) {
			return 0.0f;
		}

		float prevCdf = (idx == 0) ? 0.0f : cdf[idx - 1];
		float pixelWeight = cdf[idx] - prevCdf;

		if (pixelWeight <= 0.0f) {
			return 0.0f;
		}

		//float pixelPmf = pixelWeight / weightSum;
		//float pixelArea = 1 / (width * height);
		//float pdfUV = pixelPmf / pixelArea;
		float pixelPmf = pixelWeight / weightSum;
		float pdfUV = pixelPmf * float(width * height);

		return pdfUV / (2.0f * M_PI * M_PI * sinTheta);
	}
};