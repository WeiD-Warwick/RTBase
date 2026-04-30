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
		// Monte Carlo 87
		// r1 ​= CDF(theta) = 1 - cosTheta
		// r2 = phi / (2 * PI)

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
		/* Inverse CDF */
		// Monte Carlo 107
		// r1 ​= CDF(theta) = sin^2(theta)
		// r2 = phi / (2 * PI)

		float theta = asinf(sqrtf(r1));
		float phi = 2.0f * M_PI * r2;
		return SphericalCoordinates::sphericalToWorld(theta, phi);


		/* Malley's Method */
		// Monte Carlo 125

		// sample disk
		//float r = sqrtf(r1);
		//float phi = 2.0f * M_PI * r2;
		//float x = r * cosf(phi);
		//float y = r * sinf(phi);
		// project to hemisphere
		//float z = sqrtf(std::max(0.0f, 1.0f - x * x - y * y));

		//return Vec3(x, y, z);
	}
	static float cosineHemispherePDF(const Vec3 wi)
	{
		// cosTheta / PI
		if (wi.z <= 0.0f) return 0.0f;
		return wi.z / M_PI;
	}
	static Vec3 uniformSampleSphere(float r1, float r2)
	{
		// Monte Carlo 97
		// r1 ​= CDF(theta) = (1−cosTheta​)/2
		// r2 = CDF(phi) = Phi / (2 * PI)

		float cosTheta = 1.0f - 2.0f * r1;
		float theta = acosf(cosTheta);
		float phi = 2.0f * M_PI * r2;

		return SphericalCoordinates::sphericalToWorld(theta, phi);
	}
	static float uniformSpherePDF(const Vec3& wi)
	{
		return 1.0f / (4.0f * M_PI);
	}

	static Vec3 sampleGGXNDF(const Vec3& wo, float alpha, float r1, float r2) {
		float a = std::max(alpha, 0.001f);
		float phi = 2.0f * M_PI * r1;
		float tan2Theta = (a * a) * r2 / std::max(1.0f - r2, 1e-6f);
		float cosTheta = 1.0f / sqrtf(1.0f + tan2Theta);
		float sinTheta = sqrtf(std::max(0.0f, 1.0f - cosTheta * cosTheta));
		return Vec3(cosf(phi) * sinTheta, sinf(phi) * sinTheta, cosTheta);
	}
};