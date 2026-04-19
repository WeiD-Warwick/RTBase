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

	Colour computeDirect(ShadingData shadingData, Sampler* sampler) {

		// Compute direct lighting here

		// Is surface is specular we cannot computing direct lighting
		if (shadingData.bsdf->isPureSpecular() == true) {
			return Colour(0.0f, 0.0f, 0.0f);
		}

		float pmf = 0.f;
		Light* light = scene->sampleLight(sampler, pmf);

		if (light == NULL || pmf <= 0.0f) {
			return Colour(0.0f, 0.0f, 0.0f);
		}

		if (light->isArea()) {
			Colour emittedColour;
			float lightPdf = 0.0f;
			Vec3 p_light = light->sample(shadingData, sampler, emittedColour, lightPdf);

			if (lightPdf <= 0.0f) return Colour(0.0f, 0.0f, 0.0f);

			Vec3 shadowRayDir = p_light - shadingData.x;
			float r2 = shadowRayDir.lengthSq();
			float r = sqrtf(r2);
			Vec3 wi = shadowRayDir / r;

			// normal offset to fix self-occlusion
			Vec3 offsetOrigin = shadingData.x + shadingData.sNormal * 1e-4f;

			if (!scene->visible(offsetOrigin, p_light)) {
				return Colour(0.0f, 0.0f, 0.0f);
			}

			float cosTheta = std::max(0.0f, Dot(shadingData.sNormal, wi));

			Vec3 lightNormal = light->normal(shadingData, -wi);
			float cosThetaPrime = std::max(0.0f, Dot(lightNormal, -wi));

			if (cosTheta <= 0.0f || cosThetaPrime <= 0.0f) {
				return Colour(0.0f, 0.0f, 0.0f);
			}

			float G = (cosTheta * cosThetaPrime) / r2;
			Colour f_r = shadingData.bsdf->evaluate(shadingData, wi);

			return (emittedColour * f_r * G) / (lightPdf * pmf);
		} else {
			return Colour(0.0f, 0.0f, 0.0f);
		}
	}

	Colour pathTrace(Ray& r, Colour& pathThroughput, int depth, Sampler* sampler) {
		// Add pathtracer code here
		int maxDepth = 4;

		if (depth >= maxDepth) {
			return Colour(0, 0, 0);
		}

		IntersectionData intersection = scene->traverse(r);
		ShadingData shadingData = scene->calculateShadingData(intersection, r);

		if (shadingData.t >= FLT_MAX) {
			return pathThroughput * scene->background->evaluate(r.dir);
		}

		if (shadingData.bsdf->isLight()) {
			if (depth == 0) {
				return pathThroughput * shadingData.bsdf->emit(shadingData, shadingData.wo);
			}
			return Colour(0, 0, 0);
		}
		
		Colour Lo = pathThroughput * computeDirect(shadingData, sampler);

		Colour reflectedColour;
		float pdf = 0;
		Vec3 wi = shadingData.bsdf->sample(shadingData, sampler, reflectedColour, pdf);

		if (pdf <= 0) return Lo;

		float cosTheta = std::max(0.0f, Dot(shadingData.sNormal, wi));
		if (cosTheta <= 0.0f) return Lo;

		Colour nextPaththroughput = pathThroughput * reflectedColour * (cosTheta / pdf);
		if (nextPaththroughput.Lum() <= 0.0f) return Lo;


		Ray nextRay(shadingData.x + shadingData.sNormal * EPSILON, wi);
		Lo = Lo + pathTrace(nextRay, nextPaththroughput, depth + 1, sampler);

		return Lo;
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
		int rowsPerThread = film->height / numProcs;
		std::atomic<int> nextRow(0);

		for (int i = 0; i < numProcs; i++) {
			threads[i] = new std::thread([this, i, &nextRow]() {
				while (true) {
					int y = nextRow.fetch_add(1);
					if (y >= film->height) break;

					for (int x = 0; x < film->width; x++) {
						float px = x + samplers[i].next();
						float py = y + samplers[i].next();

						Ray ray = scene->camera.generateRay(px, py);
						//Colour col = direct(ray, &samplers[i]);

						Colour pathThroughput = Colour(1, 1, 1);
						Colour col = pathTrace(ray, pathThroughput, 0, &samplers[i]);

						film->splat(px, py, col);
						unsigned char r, g, b;
						film->tonemap(x, y, r, g, b);
						canvas->draw(x, y, r, g, b);
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
