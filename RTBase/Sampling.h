#pragma once

#include "Core.h"
#include <random>
#include <algorithm>

class Sampler
{
public:
	virtual float next() = 0;
};

class MTRandom : public Sampler
{
public:
	std::mt19937 generator;
	std::uniform_real_distribution<float> dist;
	MTRandom(unsigned int seed = 1) : dist(0.0f, 1.0f)
	{
		generator.seed(seed);
	}
	float next()
	{
		return dist(generator);
	}
};

// Note all of these distributions assume z-up coordinate system
class SamplingDistributions
{
public:
	static Vec3 uniformSampleHemisphere(float r1, float r2)
	{
		float cosTheta = 1.0f - r1;
		float theta = acosf(cosTheta);
		float phi = 2.0f * M_PI * r2;

		return SphericalCoordinates::sphericalToWorld(theta, phi);
	}
	static float uniformHemispherePDF(const Vec3 wi)
	{
		if (wi.z <= 0.0f) return 0.0f;

		return 1.0f / (2.0f * M_PI);
	}
	static Vec3 cosineSampleHemisphere(float r1, float r2)
	{

		float theta = acosf(sqrtf(r1));
		float phi = 2.0f * M_PI * r2;
		return SphericalCoordinates::sphericalToWorld(theta, phi);

	}
	static float cosineHemispherePDF(const Vec3 wi)
	{
		// cosTheta / PI
		if (wi.z <= 0.0f) return 0.0f;
		return wi.z / M_PI;
	}
	static Vec3 uniformSampleSphere(float r1, float r2)
	{
		float cosTheta = 1.0f - 2.0f * r1;
		float theta = acosf(cosTheta);
		float phi = 2.0f * M_PI * r2;

		return SphericalCoordinates::sphericalToWorld(theta, phi);
	}
	static float uniformSpherePDF(const Vec3& wi)
	{
		return 1.0f / (4.0f * M_PI);
	}

	static Vec3 sampleGGX(const Vec3& wo, float alpha, float r1, float r2) {
		float alphaSq = alpha * alpha;
		float cosTheta = std::sqrt((1.0f - r1) / (r1 * (alphaSq - 1.0f) + 1.0f));
		float sinTheta = std::sqrt(std::max(0.0f, 1.0f - cosTheta * cosTheta));

		float phi = 2.0f * M_PI * r2;

		Vec3 wiLocal(sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta);
		return wiLocal;
	}
};