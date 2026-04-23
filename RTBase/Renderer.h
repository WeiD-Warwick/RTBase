#pragma once

#include "Core.h"
#include "Sampling.h"
#include "Geometry.h"
#include "Imaging.h"
#include "Materials.h"
#include "Lights.h"
#include "Scene.h"
#include "GamesEngineeringBase.h"
#include <thread>
#include <functional>
#include <atomic>

class RayTracer
{
public:
	Scene* scene;
	GamesEngineeringBase::Window* canvas;
	Film* film;
	MTRandom *samplers;
	std::thread **threads;
	int numProcs;
	void init(Scene* _scene, GamesEngineeringBase::Window* _canvas)
	{
		scene = _scene;
		canvas = _canvas;
		film = new Film();
		film->init((unsigned int)scene->camera.width, (unsigned int)scene->camera.height, new BoxFilter());
		SYSTEM_INFO sysInfo;
		GetSystemInfo(&sysInfo);
		numProcs = sysInfo.dwNumberOfProcessors;
		threads = new std::thread*[numProcs];
		samplers = new MTRandom[numProcs];

		for (int i = 0; i < numProcs; i++) samplers[i].generator.seed(i + 1);
		clear();
	}

	void clear()
	{
		film->clear();
	}

	// NEE
	Colour computeDirect(ShadingData shadingData, Sampler* sampler) {
		// Only non-specular surfaces can connect
		if (shadingData.bsdf->isPureSpecular() == true) {
			return Colour(0.0f, 0.0f, 0.0f);
		}

		// Light Transport 159
		// sample light
		float LightPmf = 0.f;
		Light* light = scene->sampleLight(sampler, LightPmf);

		if (light == NULL || LightPmf <= 0.0f) {
			return Colour(0.0f, 0.0f, 0.0f);
		}

		float lightPdf = 0.0f;
		auto lightSampleValue = light->sample(shadingData, sampler, lightPdf);
		if (lightPdf <= 0.0f) return Colour(0.0f, 0.0f, 0.0f);
		// total sample light pdf
		lightPdf *= LightPmf;
		if (lightPdf <= 0.0f) return Colour(0.0f, 0.0f, 0.0f);

		// Area Angle Sampling
		if (light->isArea()) {
			// Sample point on light and store returned emission
			Vec3 sampleLightPoint = lightSampleValue;
			Vec3 shadowRayDir = sampleLightPoint - shadingData.x;
			Vec3 wi = shadowRayDir.normalize();
			Colour emission = light->evaluate(wi);
			
			// Calculate visibility
			bool visible = scene->visible(shadingData.x, sampleLightPoint);
			if (!visible) return Colour(0.0f, 0.0f, 0.0f);

			// Calculate Geometry Term (Mento Carlo 65)
			Vec3 lightNormal = light->normal(shadingData, -wi);
			float cosTheta = std::max(Dot(wi, shadingData.sNormal), 0.0f);
			if (cosTheta <= 0.0f) return Colour(0.0f, 0.0f, 0.0f);
			float cosThetaPrime = std::max(Dot(-wi, lightNormal), 0.0f);
			if (cosThetaPrime <= 0.0f) return Colour(0.0f, 0.0f, 0.0f);

			float G = cosTheta * cosThetaPrime / shadowRayDir.lengthSq();

			// Evaluate BSDF
			Colour reflectedColour = shadingData.bsdf->evaluate(shadingData, wi);
			float bsdfPdf = std::max(shadingData.bsdf->PDF(shadingData, wi), 0.0f);
			float bsdfPdfArea = bsdfPdf * cosThetaPrime / shadowRayDir.lengthSq();

			// MIS Weight
			float weight = powerHeuristic(lightPdf, bsdfPdfArea);

			return (emission * reflectedColour * G * weight) / lightPdf;
		} else {
			Vec3 shadowRayDir = lightSampleValue;

			Vec3 wi = shadowRayDir.normalize();
			Colour emission = light->evaluate(wi);

			// Evaluate visibility to outside scene bounds
			Ray shadowRay(shadingData.x, wi);
			IntersectionData shadowHit = scene->traverse(shadowRay);
			if (shadowHit.t < FLT_MAX) return Colour(0.0f, 0.0f, 0.0f);

			// Evaluate Geometry Term for environment maps 
			float cosTheta = std::max(Dot(wi, shadingData.sNormal), 0.0f);
			if (cosTheta <= 0.0f) return Colour(0.0f, 0.0f, 0.0f);

			// Evaluate BSDF and multiply terms and return 
			Colour reflectedColour = shadingData.bsdf->evaluate(shadingData, wi);
			float bsdfPdf = shadingData.bsdf->PDF(shadingData, wi);

			// MIS Weight
			float weight = powerHeuristic(lightPdf, bsdfPdf);

			return (emission * reflectedColour * cosTheta * weight) / lightPdf;
		}
	}

	float lightPdfFromPrevPoint(const ShadingData& prevShadingData, const ShadingData& shadingData, const Vec3& wi, int hitTriangleID) {
		if (scene->lights.empty()) return 0.0f;

		float lightPmf = scene->lightPMF();

		if (shadingData.t >= FLT_MAX) {
			return scene->background->PDF(prevShadingData, wi) * lightPmf;
		}

		if (!shadingData.bsdf->isLight()) return 0.0f;

		if (hitTriangleID < 0 || hitTriangleID >= (int)scene->triangles.size()) return 0.0f;

		Triangle* hitTriangle = &scene->triangles[hitTriangleID];

		AreaLight tempLight;
		tempLight.triangle = hitTriangle;
		float pdfArea = tempLight.PDF(prevShadingData, wi);
		if (pdfArea <= 0.0f) return 0.0f;

		Vec3 lightNormal = hitTriangle->gNormal();
		float dist2 = (shadingData.x - prevShadingData.x).lengthSq();
		float cosThetaPrime = std::max(Dot(-wi, lightNormal), 0.0f);
		if (cosThetaPrime <= 0.0f) return 0.0f;

		float pdfOmega = pdfArea * dist2 / cosThetaPrime;
		return pdfOmega * lightPmf;
	}

	Colour pathTrace(Ray& r, Colour& pathThroughput, int depth, Sampler* sampler, const ShadingData* prevShadingData, float prevBsdfPdf) {
		// Get Shading Point
		IntersectionData intersection = scene->traverse(r);
		ShadingData shadingData = scene->calculateShadingData(intersection, r);

		// non intersection
		if (shadingData.t >= FLT_MAX) {
			Colour Le = scene->background->evaluate(r.dir);
			// First time or from specular, return Le
			if (depth == 0 || prevShadingData->bsdf->isPureSpecular()) {
				return pathThroughput * Le;
			}
			float lightPdf = lightPdfFromPrevPoint(*prevShadingData, shadingData, r.dir, -1);

			float weight = powerHeuristic(prevBsdfPdf, lightPdf);
			return pathThroughput * Le * weight;
		}

		// hit light
		if (shadingData.bsdf->isLight()) {
			if (depth == 0) {
				return pathThroughput * shadingData.bsdf->emit(shadingData, shadingData.wo);
			}
			if (Dot(r.dir, shadingData.gNormal) < 0.0f) {
				return Colour(0.0f, 0.0f, 0.0f);
			}
			Colour Le = shadingData.bsdf->emit(shadingData, shadingData.wo);
			if (prevShadingData->bsdf->isPureSpecular()) {
				return pathThroughput * Le;
			}
			float lightPdf = lightPdfFromPrevPoint(*prevShadingData, shadingData, r.dir, intersection.ID);
			float weight = powerHeuristic(prevBsdfPdf, lightPdf);
			return pathThroughput * Le * weight;
		}
		else {
			// Direct Light (NEE)
			Colour Lo = pathThroughput * computeDirect(shadingData, sampler);

			// Indirect Light (traverse)
			float pdf = 0.0f;
			Vec3 wi = shadingData.bsdf->sample(shadingData, sampler, pdf);
			if (pdf <= 0) return Lo;

			Colour reflectedColour = shadingData.bsdf->evaluate(shadingData, wi);

			float cosTheta = std::max(Dot(wi, shadingData.sNormal), 0.0f);
			if (cosTheta <= 0.0f) return Lo;

			Colour nextPaththroughput = pathThroughput * reflectedColour * cosTheta / pdf;
			if (nextPaththroughput.Lum() <= 0.0f) return Lo;

			// Russian Roulette
			// Light TracePath 199
			if (depth >= 3) {
				float continueProbability = std::min(nextPaththroughput.Lum(), 0.95f);
				if (continueProbability <= 0.0f || sampler->next() > continueProbability) return Lo;
				nextPaththroughput = nextPaththroughput / continueProbability;
			}

			Ray nextRay(shadingData.x, wi);
			return Lo + pathTrace(nextRay, nextPaththroughput, depth + 1, sampler, &shadingData, pdf);
		}
	}

	float powerHeuristic(float pdfA, float pdfB) {
		float a2 = pdfA * pdfA;
		float b2 = pdfB * pdfB;
		return a2 / (a2 + b2);
	}

	Colour direct(Ray& r, Sampler* sampler) {
		IntersectionData intersection = scene->traverse(r);
		ShadingData shadingData = scene->calculateShadingData(intersection, r);
		if (shadingData.t < FLT_MAX) {
			if (shadingData.bsdf->isLight()) {
				return shadingData.bsdf->emit(shadingData, shadingData.wo);
			}
			return computeDirect(shadingData, sampler);
		}
		return scene->background->evaluate(r.dir);
	}

	Colour albedo(Ray& r) {
		IntersectionData intersection = scene->traverse(r);
		ShadingData shadingData = scene->calculateShadingData(intersection, r);
		if (shadingData.t < FLT_MAX)
		{
			if (shadingData.bsdf->isLight())
			{
				return shadingData.bsdf->emit(shadingData, shadingData.wo);
			}
			return shadingData.bsdf->evaluate(shadingData, Vec3(0, 1, 0));
		}
		return scene->background->evaluate(r.dir);
	}

	Colour viewNormals(Ray& r)
	{
		IntersectionData intersection = scene->traverse(r);
		if (intersection.t < FLT_MAX)
		{
			ShadingData shadingData = scene->calculateShadingData(intersection, r);
			return Colour(fabsf(shadingData.sNormal.x), fabsf(shadingData.sNormal.y), fabsf(shadingData.sNormal.z));
		}
		return Colour(0.0f, 0.0f, 0.0f);
	}

	void render() {
		film->incrementSPP();
		std::atomic<int> nextTile(0);
		const int tileSize = 16;
		const int tilesX = (film->width + tileSize - 1) / tileSize;
		const int tilesY = (film->height + tileSize - 1) / tileSize;
		const int totalTiles = tilesX * tilesY;

		for (int i = 0; i < numProcs; i++) {
			threads[i] = new std::thread([this, i, &nextTile, tileSize, tilesX, tilesY, totalTiles]() {
				while (true) {
					int tileIndex = nextTile.fetch_add(1);
					if (tileIndex >= totalTiles) break;
					int tileY = tileIndex / tilesX;
					int tileX = tileIndex % tilesX;
					int xStart = tileX * tileSize;
					int yStart = tileY * tileSize;
					int xEnd = std::min((int)film->width, xStart + tileSize);
					int yEnd = std::min((int)film->height, yStart + tileSize);

					for (int y = yStart; y < yEnd; y++) {
						for (int x = xStart; x < xEnd; x++) {
							float px = x + samplers[i].next();
							float py = y + samplers[i].next();

							Ray ray = scene->camera.generateRay(px, py);
							//Colour col = direct(ray, &samplers[i]);

							Colour pathThroughput = Colour(1, 1, 1);
							Colour col = pathTrace(ray, pathThroughput, 0, &samplers[i], nullptr, 0.0f);

							film->splat(px, py, col);
							unsigned char r, g, b;
							film->tonemap(x, y, r, g, b);
							canvas->draw(x, y, r, g, b);
						}
					}
				}
				});
		}

		for (int i = 0; i < numProcs; i++) {
			threads[i]->join();
			delete threads[i];
		}
	}

	int getSPP()
	{
		return film->SPP;
	}
	void saveHDR(std::string filename)
	{
		film->save(filename);
	}
	void savePNG(std::string filename)
	{
		stbi_write_png(filename.c_str(), canvas->getWidth(), canvas->getHeight(), 3, canvas->getBackBuffer(), canvas->getWidth() * 3);
	}
};
