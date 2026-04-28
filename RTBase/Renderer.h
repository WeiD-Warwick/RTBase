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
#include <cmath>

class RayTracer
{
public:
	enum class RenderMode {
		PathTrace,
		DebugSNormal,
		DebugGNormal,
		DebugNormalDelta,
	};

	Scene* scene;
	GamesEngineeringBase::Window* canvas;
	Film* film;
	MTRandom *samplers;
	std::thread **threads;
	int numProcs;
	RenderMode renderMode = RenderMode::PathTrace;
	float fireflyClamp = 10.0f;

	void init(Scene* _scene, GamesEngineeringBase::Window* _canvas)
	{
		scene = _scene;
		canvas = _canvas;
		film = new Film();
		film->init((unsigned int)scene->camera.width, (unsigned int)scene->camera.height, new GaussianFilter());
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
	void setRenderMode(RenderMode mode) {
		if (renderMode != mode) {
			renderMode = mode;
			clear();
		}
	}
	RenderMode getRenderMode() const {
		return renderMode;
	}

	float balanceHeuristic(float pdfA, float pdfB) {
		float sum = pdfA + pdfB;
		if (sum <= 0.0f) return 0.0f;
		return pdfA / sum;
	}

	float powerHeuristic(float pdfA, float pdfB) {
		float a2 = pdfA * pdfA;
		float b2 = pdfB * pdfB;
		float sum = a2 + b2;
		if (sum <= 0.0f) return 0.0f;
		return a2 / sum;
	}

	Colour clampSample(Colour colour) {
		if (!std::isfinite(colour.r) || !std::isfinite(colour.g) || !std::isfinite(colour.b)) {
			return Colour(0.0f, 0.0f, 0.0f);
		}

		colour.r = std::max(colour.r, 0.0f);
		colour.g = std::max(colour.g, 0.0f);
		colour.b = std::max(colour.b, 0.0f);

		float lum = colour.Lum();
		if (!std::isfinite(lum) || lum <= 0.0f || fireflyClamp <= 0.0f) {
			return colour;
		}

		if (lum > fireflyClamp) {
			colour = colour * (fireflyClamp / lum);
		}
		return colour;
	}

	Vec3 offsetRayOrigin(const Vec3& x, const Vec3& gNormal, const Vec3& w) {
		float side = Dot(w, gNormal) >= 0.0f ? 1.0f : -1.0f;
		return x + gNormal * (EPSILON * side);
	}

	// NEE (pdf base Light Area)
	Colour computeDirect(ShadingData shadingData, Sampler* sampler) {
		// Is surface is specular we cannot computing direct lighting
		if (shadingData.bsdf->isPureSpecular() == true) {
			return Colour(0.0f, 0.0f, 0.0f);
		}

		float LightPmf = 0.0f;
		Light* light = scene->sampleLight(sampler, LightPmf);
		if (light == NULL || LightPmf <= 0.0f) return Colour(0.0f, 0.0f, 0.0f);

		if (light->isArea()) {
			// Sample point on light
			Colour emission;
			float lightPdf = 0.0f;
			Vec3 sampleLightPoint = light->sample(shadingData, sampler, emission, lightPdf);
			if (lightPdf <= 0.0f) return Colour(0.0f, 0.0f, 0.0f);
			// total sample light pdf (area pdf)
			lightPdf *= LightPmf;

			Vec3 shadowRayDir = sampleLightPoint - shadingData.x;
			Vec3 wi = shadowRayDir.normalize();

			// Calculate visibility
			bool visible = scene->visible(shadingData.x, sampleLightPoint);
			if (!visible) return Colour(0.0f, 0.0f, 0.0f);

			// Calculate Geometry Term (Mento Carlo 65)
			
			float cosTheta = Dot(wi, shadingData.gNormal);
			if (cosTheta <= 0.0f) return Colour(0.0f, 0.0f, 0.0f);
			float absCosTheta = fabsf(cosTheta);

			Vec3 lightNormal = light->normal(shadingData, wi);
			float cosThetaPrime = Dot(-wi, lightNormal);
			if (cosThetaPrime <= 0.0f) return Colour(0.0f, 0.0f, 0.0f);

			float G = absCosTheta * cosThetaPrime / shadowRayDir.lengthSq();

			// Evaluate BSDF
			Colour reflectedColour = shadingData.bsdf->evaluate(shadingData, wi);
			float bsdfPdf = shadingData.bsdf->PDF(shadingData, wi);
			// solid to area measure
			float bsdfPdfArea = bsdfPdf * cosThetaPrime / shadowRayDir.lengthSq();

			// MIS Weight
			float weight = powerHeuristic(lightPdf, bsdfPdfArea);

			return (emission * reflectedColour * G * weight) / lightPdf;
		} else {
			Colour envColour;
			float lightPdf = 0.0f;
			Vec3 shadowRayDir = light->sample(shadingData, sampler, envColour, lightPdf);
			if (lightPdf <= 0.0f) return Colour(0.0f, 0.0f, 0.0f);

			lightPdf *= LightPmf;

			Vec3 wi = shadowRayDir.normalize();

			// Evaluate visibility to outside scene bounds
			Ray shadowRay(offsetRayOrigin(shadingData.x, shadingData.gNormal, wi), wi);
			IntersectionData shadowHit = scene->traverse(shadowRay);
			if (shadowHit.t < FLT_MAX) return Colour(0.0f, 0.0f, 0.0f);

			// Evaluate Geometry Term for environment maps 
			float cosTheta = Dot(wi, shadingData.gNormal);
			if (!shadingData.bsdf->isTwoSided() && cosTheta <= 0.0f) return Colour(0.0f, 0.0f, 0.0f);
			float absCosTheta = fabsf(cosTheta);

			// Evaluate BSDF and multiply terms and return 
			Colour reflectedColour = shadingData.bsdf->evaluate(shadingData, wi);
			float bsdfPdf = shadingData.bsdf->PDF(shadingData, wi);

			// MIS Weight
			float weight = powerHeuristic(lightPdf, bsdfPdf);
			return (envColour * reflectedColour * absCosTheta * weight) / lightPdf;
		}
	}

	float lightPdfFromPrevPoint(const ShadingData& prevShadingData, const ShadingData& shadingData, const Vec3& wi, int hitTriangleID) {

		if (hitTriangleID >= 0 && hitTriangleID < scene->triangles.size()) {
			float lightPmf = 0.0f;
			Triangle* hitTriangle = &scene->triangles[hitTriangleID];
			for (int i = 0; i < scene->lights.size(); i++) {
				if (!scene->lights[i]->isArea()) continue;
				AreaLight* area = dynamic_cast<AreaLight*>(scene->lights[i]);
				if (area != NULL && area->triangle == hitTriangle) {
					lightPmf = scene->lightSelectionPMF();
					break;
				}
			}
			if (lightPmf <= 0.0f) return 0.0f;

			if (hitTriangle->area <= 0.0f) return 0.0f;

			float pdfArea = 1.0f / hitTriangle->area;
			Vec3 lightNormal = hitTriangle->gNormal();
			float dist2 = (shadingData.x - prevShadingData.x).lengthSq();
			float cosThetaPrime = std::max(Dot(-wi, lightNormal), 0.0f);
			if (cosThetaPrime <= 0.0f) return 0.0f;

			float pdfW = pdfArea * dist2 / cosThetaPrime;
			return pdfW * lightPmf;
		}

		// Background / environment light
		if (scene->background != NULL) {
			float lightPmf = 0.0f;
			bool backgroundIsSampledLight = false;
			for (int i = 0; i < scene->lights.size(); i++) {
				if (scene->lights[i] == scene->background) {
					backgroundIsSampledLight = true;
					lightPmf = scene->lightSelectionPMF();
					break;
				}
			}
			if (!backgroundIsSampledLight) return 0.0f;

			float bgPdf = scene->background->PDF(prevShadingData, wi);
			if (bgPdf <= 0.0f) return 0.0f;
			return bgPdf * lightPmf;
		}
		return 0.0f;
	}

	Colour pathTrace(Ray& r, Colour pathThroughput, int depth, Sampler* sampler, ShadingData* prevShadingData, float prevBsdfPdf) {
		int maxDepth = 8;

		IntersectionData intersection = scene->traverse(r);
		ShadingData shadingData = scene->calculateShadingData(intersection, r);

		// Hit background
		if (shadingData.t >= FLT_MAX) {
			Colour Le = scene->background->evaluate(r.dir);
			// First time or from specular, return Le
			if (depth == 0 || prevShadingData->bsdf->isPureSpecular()) {
				return pathThroughput * Le;
			}
			float lightPdfW = lightPdfFromPrevPoint(*prevShadingData, shadingData, r.dir, -1);
			float weight = powerHeuristic(prevBsdfPdf, lightPdfW);
			return pathThroughput * Le * weight;
		}

		// hit light
		if (shadingData.bsdf->isLight()) {
			Vec3 lightNormal = scene->triangles[intersection.ID].gNormal();
			if (Dot(r.dir, lightNormal) > 0.0f) {
				return Colour(0.0f, 0.0f, 0.0f);
			}

			Colour Le = shadingData.bsdf->emit(shadingData, shadingData.wo);
			if (depth == 0 || prevShadingData->bsdf->isPureSpecular()) {
				return pathThroughput * Le;
			}
			float lightPdfW = lightPdfFromPrevPoint(*prevShadingData, shadingData, r.dir, intersection.ID);
			float weight = powerHeuristic(prevBsdfPdf, lightPdfW);
			return pathThroughput * Le * weight;
		}
		else {
			// Direct Light (NEE)
			Colour Lo = pathThroughput * computeDirect(shadingData, sampler);

			if (depth >= maxDepth) {
				return Lo;
			}

			// Indirect Light (traverse)
			float pdf = 0.0f;
			Colour reflectedColour;
			Vec3 wi = shadingData.bsdf->sample(shadingData, sampler, reflectedColour, pdf);
			if (pdf <= 0) return Lo;

			float cosTheta = Dot(wi, shadingData.gNormal);
			if (!shadingData.bsdf->isTwoSided() && cosTheta <= 0.0f) return Lo;
			float absCosTheta = fabsf(cosTheta);

			Colour nextPaththroughput = pathThroughput * reflectedColour * absCosTheta / pdf;
			if (nextPaththroughput.Lum() <= 0.0f) return Lo;

			// Russian Roulette
			// Light TracePath 199
			if (depth >= 3) {
				float continueProbability = std::min(nextPaththroughput.Lum(), 0.95f);
				if (continueProbability <= 0.0f || sampler->next() > continueProbability) return Lo;
				nextPaththroughput = nextPaththroughput / continueProbability;
			}

			Ray nextRay(offsetRayOrigin(shadingData.x, shadingData.gNormal, wi), wi);
			return Lo + pathTrace(nextRay, nextPaththroughput, depth + 1, sampler, &shadingData, pdf);
		}
	}

	Colour direct(Ray& r, Sampler* sampler)
	{
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

	Colour viewNormals(Ray& r) {
		IntersectionData intersection = scene->traverse(r);
		if (intersection.t < FLT_MAX) {
			ShadingData shadingData = scene->calculateShadingData(intersection, r);
			return Colour(fabsf(shadingData.sNormal.x), fabsf(shadingData.sNormal.y), fabsf(shadingData.sNormal.z));
		}
		return Colour(0.0f, 0.0f, 0.0f);
	}

	Colour viewGNormals(Ray& r) {
		IntersectionData intersection = scene->traverse(r);
		if (intersection.t < FLT_MAX) {
			ShadingData shadingData = scene->calculateShadingData(intersection, r);
			return Colour(fabsf(shadingData.gNormal.x), fabsf(shadingData.gNormal.y), fabsf(shadingData.gNormal.z));
		}
		return Colour(0.0f, 0.0f, 0.0f);
	}
	Colour viewNormalDelta(Ray& r) {
		IntersectionData intersection = scene->traverse(r);
		if (intersection.t < FLT_MAX) {
			ShadingData shadingData = scene->calculateShadingData(intersection, r);
			Vec3 d = shadingData.sNormal - shadingData.gNormal;
			return Colour(fabsf(d.x), fabsf(d.y), fabsf(d.z));
		}
		return Colour(0.0f, 0.0f, 0.0f);
	}
	void render()
	{
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
							Colour pathThroughput = Colour(1.0f, 1.0f, 1.0f);
							Colour col;

							switch (renderMode) {
							case RayTracer::RenderMode::PathTrace:
								col = pathTrace(ray, pathThroughput, 0, &samplers[i], nullptr, 0.0f);
								break;
							case RayTracer::RenderMode::DebugSNormal:
								col = viewNormals(ray);
								break;
							case RayTracer::RenderMode::DebugGNormal:
								col = viewGNormals(ray);
								break;
							case RayTracer::RenderMode::DebugNormalDelta:
								col = viewNormalDelta(ray);
								break;
							default:
								col = pathTrace(ray, pathThroughput, 0, &samplers[i], nullptr, 0.0f);
								break;
							}

							col = clampSample(col);
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
