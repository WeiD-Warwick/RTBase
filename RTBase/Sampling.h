#pragma once

#include "Core.h"
#include <random>
#include <algorithm>

class Sampler {
public:
	virtual float next() = 0;
};

// Mersenne Twister
class MTRandom : public Sampler {
public:
	std::mt19937 generator;
	std::uniform_real_distribution<float> dist;
	MTRandom(unsigned int seed = 1) : dist(0.0f, 1.0f) {
		generator.seed(seed);
	}

	float next() {
		return dist(generator);
	}
};

// Note all of these distributions assume z-up coordinate system
class SamplingDistributions {

public:

	static Vec3 uniformSampleHemisphere(float r1, float r2) {

		float cosTheta = 1.0f - r1;
		float theta = acosf(cosTheta);
		float phi = 2.0f * M_PI * r2;

		return SphericalCoordinates::sphericalToWorld(theta, phi);
	}

	static float uniformHemispherePDF(const Vec3 wi) {

		if (wi.z <= 0.0f) return 0.0f;

		return 1.0f / (2.0f * M_PI);
	}

	static Vec3 cosineSampleHemisphere(float r1, float r2) {

		float cosTheta = sqrtf(1.0f - r1);
		float theta = acosf(cosTheta);
		float phi = 2.0f * M_PI * r2;

		return SphericalCoordinates::sphericalToWorld(theta, phi);
	}

	static float cosineHemispherePDF(const Vec3 wi) {

		if (wi.z <= 0.0f) return 0.0f;

		return wi.z / M_PI;
	}

	static Vec3 uniformSampleSphere(float r1, float r2) {

		float cosTheta = 1.0f - 2.0f * r1;
		float theta = acosf(cosTheta);

		float phi = 2.0f * M_PI * r2;

		return SphericalCoordinates::sphericalToWorld(theta, phi);
	}

	static float uniformSpherePDF(const Vec3& wi) {
		return 1.0f / (4.0f * M_PI);
	}
};