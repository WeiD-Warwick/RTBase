

#include "GEMLoader.h"
#include "Renderer.h"
#include "SceneLoader.h"
#define NOMINMAX
#include "GamesEngineeringBase.h"
#include <unordered_map>

std::string renderModeName(RayTracer::RenderMode renderMode) {
	switch (renderMode) {
	case RayTracer::RenderMode::InstantRadiosity:
		return "InstantRadiosity";
	case RayTracer::RenderMode::LightTrace:
		return "LightTrace";
	default:
		return "PathTrace";
	}
}

RayTracer::RenderMode parseRenderMode(const std::string& name) {
	if (name == "pt") return RayTracer::RenderMode::PathTrace;
	if (name == "ir") return RayTracer::RenderMode::InstantRadiosity;
	if (name == "lt") return RayTracer::RenderMode::LightTrace;
	std::cerr << "Warning: Unknown render mode '" << name << "'\n";
	return RayTracer::RenderMode::PathTrace;
}

std::string outputBaseName(const std::string& sceneName, int spp, RayTracer::RenderMode renderMode) {
	size_t pos = sceneName.find_last_of("/\\");
	std::string shortSceneName = (pos == std::string::npos) ? sceneName : sceneName.substr(pos + 1);
	std::string runName = shortSceneName + "-" + renderModeName(renderMode) + "-" + std::to_string(spp);
	return "results\\" + runName;
}

void runTests()
{
	// Plane test
	Ray ray(Vec3(0, 1, 0), Vec3(0, -1, 0));
	Plane plane;
	Vec3 normal(0, 1, 0);
	plane.init(normal, 0);

	float t;
	bool hit = plane.rayIntersect(ray, t);

	std::cout << "Plane hit: " << hit << ", t: " << t << std::endl;

	Ray ray2(Vec3(0, 1, 0), Vec3(0, 1, 0));
	float t2;
	bool hit2 = plane.rayIntersect(ray2, t2);

	std::cout << "Plane hit (miss test): " << hit2 << ", t: " << t2 << std::endl;

	// Triangle test
	Vertex v0, v1, v2;
	v0.p = Vec3(0, 0, 0);
	v1.p = Vec3(1, 0, 0);
	v2.p = Vec3(0, 1, 0);

	Triangle tri;
	tri.init(v0, v1, v2, 0);

	Ray ray3(Vec3(0.2f, 0.2f, 1), Vec3(0, 0, -1));

	float t3, u, v;
	bool hit3 = tri.rayIntersect(ray3, t3, u, v);

	std::cout << "Triangle hit: " << hit3 << ", t: " << t3 << " u: " << u << " v: " << v << std::endl;
}

int main(int argc, char* argv[])
{
	// Add call to tests if required
	// runTests();

	// Initialize default parameters
	std::string sceneName = "Assets/cornell-box";
	//std::string sceneName = "Assets/MaterialsScene";
	//std::string sceneName = "Assets/bathroom";
	//std::string sceneName = "Assets/bathroom2";
	//std::string sceneName = "Assets/living-room";
	//std::string sceneName = "Assets/living-room-2";
	//std::string sceneName = "Assets/living-room-3";
	//std::string sceneName = "Assets/glass-of-water";
	std::string filename = "GI.hdr";
	//unsigned int SPP = 8192;
	unsigned int SPP = 32;
	//RayTracer::RenderMode renderMode = RayTracer::RenderMode::PathTrace;
	//RayTracer::RenderMode renderMode = RayTracer::RenderMode::InstantRadiosity;
	RayTracer::RenderMode renderMode = RayTracer::RenderMode::LightTrace;

	if (argc > 1)
	{
		std::unordered_map<std::string, std::string> args;
		for (int i = 1; i < argc; ++i)
		{
			std::string arg = argv[i];
			if (!arg.empty() && arg[0] == '-')
			{
				std::string argName = arg;
				if (i + 1 < argc)
				{
					std::string argValue = argv[++i];
					args[argName] = argValue;
				}
				else
				{
					std::cerr << "Error: Missing value for argument '" << arg << "'\n";
				}
			}
			else
			{
				std::cerr << "Warning: Ignoring unexpected argument '" << arg << "'\n";
			}
		}
		for (const auto& pair : args)
		{
			if (pair.first == "-scene")
			{
				sceneName = pair.second;
			}
			if (pair.first == "-outputFilename")
			{
				filename = pair.second;
			}
			if (pair.first == "-SPP")
			{
				SPP = stoi(pair.second);
			}
			if (pair.first == "-mode")
			{
				renderMode = parseRenderMode(pair.second);
			}
		}
	}
	Scene* scene = loadScene(sceneName);
	scene->build();
	GamesEngineeringBase::Window canvas;
	canvas.create((unsigned int)scene->camera.width, (unsigned int)scene->camera.height, "Tracer", false);
	RayTracer rt;
	rt.init(scene, &canvas);
	rt.setRenderMode(renderMode);
	bool running = true;
	GamesEngineeringBase::Timer timer;
	while (running)
	{
		canvas.checkInput();
		canvas.clear();
		if (canvas.keyPressed(VK_ESCAPE))
		{
			break;
		}
		if (canvas.keyPressed('W'))
		{
			viewcamera.forward();
			rt.clear();
		}
		if (canvas.keyPressed('S'))
		{
			viewcamera.back();
			rt.clear();
		}
		if (canvas.keyPressed('A'))
		{
			viewcamera.left();
			rt.clear();
		}
		if (canvas.keyPressed('D'))
		{
			viewcamera.right();
			rt.clear();
		}
		if (canvas.keyPressed('E'))
		{
			viewcamera.flyUp();
			rt.clear();
		}
		if (canvas.keyPressed('Q'))
		{
			viewcamera.flyDown();
			rt.clear();
		}

		// Time how long a render call takes
		timer.reset();
		rt.render();
		float t = timer.dt();
		// Write
		std::cout << t << std::endl;
		if (canvas.keyPressed('P'))
		{
			rt.saveHDR(filename);
		}
		if (canvas.keyPressed('L'))
		{
			size_t pos = filename.find_last_of('.');
			std::string ldrFilename = filename.substr(0, pos) + "-" + std::to_string(rt.getSPP()) + ".png";
			rt.savePNG(ldrFilename);
		}
		if (SPP == rt.getSPP())
		{
			std::string baseName = outputBaseName(sceneName, SPP, renderMode);
			std::cout << "Saving " << baseName << " outputs..." << std::endl;
			rt.saveFinalOutputs(baseName);
			std::cout << "Done." << std::endl;
			break;
		}
		canvas.present();
	}
	return 0;
}