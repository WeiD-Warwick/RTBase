#pragma once
#define EPSILON 0.001f
#include "Core.h"
#include "Sampling.h"

class Ray
{
public:
	Vec3 o;
	Vec3 dir;
	Vec3 invDir;
	Ray()
	{
	}
	Ray(Vec3 _o, Vec3 _d)
	{
		init(_o, _d);
	}
	void init(Vec3 _o, Vec3 _d)
	{
		o = _o;
		dir = _d;
		invDir = Vec3(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);
	}
	Vec3 at(const float t) const
	{
		return (o + (dir * t));
	}

	Vec3 offsetOrigin(const Vec3& x, const Vec3& gNormal, const Vec3& wi) {
		const float side = Dot(wi, gNormal) >= 0.0f ? 1.0f : -1.0f;
		return x + gNormal * (EPSILON * side);
	}
};

class AABB
{
public:
	Vec3 max;
	Vec3 min;
	AABB()
	{
		reset();
	}
	void reset()
	{
		max = Vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
		min = Vec3(FLT_MAX, FLT_MAX, FLT_MAX);
	}
	void extend(const Vec3 p)
	{
		max = Max(max, p);
		min = Min(min, p);
	}
	void extend(const AABB aabb) {
		extend(aabb.max);
		extend(aabb.min);
	}
	bool rayAABB(const Ray& r, float& t)
	{
		Vec3 Tmin = (min - r.o) * r.invDir;
		Vec3 Tmax = (max - r.o) * r.invDir;
		Vec3 Tentry = Min(Tmin, Tmax);
		Vec3 Texit = Max(Tmin, Tmax);

		float tentry = std::max(Tentry.x, std::max(Tentry.y, Tentry.z));
		float texit = std::min(Texit.x, std::min(Texit.y, Texit.z));

		if (tentry > texit || texit < 0.0f) return false;

		t = (tentry < 0.0f) ? 0.0f : tentry;
		return true;
	}
	float area()
	{
		Vec3 size = max - min;
		return ((size.x * size.y) + (size.y * size.z) + (size.x * size.z)) * 2.0f;
	}
};

class Plane
{
public:
	Vec3 n;
	float d;
	void init(Vec3& _n, float _d)
	{
		n = _n;
		d = _d;
	}
	// Add code here
	bool rayIntersect(Ray& r, float& t)
	{
		float denom = Dot(n, r.dir);
		if (fabs(denom) < EPSILON) {
			return false;
		}

		t = (-d - Dot(n, r.o)) / denom;
		if (t < 0) return false;
		return true;
	}
};

class Triangle
{
public:
	Vertex vertices[3];
	Vec3 e1; // Edge 1
	Vec3 e2; // Edge 2
	Vec3 n; // Geometric Normal
	float area; // Triangle area
	float d; // For ray triangle if needed
	unsigned int materialIndex;
	void init(Vertex v0, Vertex v1, Vertex v2, unsigned int _materialIndex)
	{
		materialIndex = _materialIndex;
		vertices[0] = v0;
		vertices[1] = v1;
		vertices[2] = v2;
		e1 = vertices[2].p - vertices[1].p;
		e2 = vertices[0].p - vertices[2].p;
		n = e1.cross(e2).normalize();
		area = e1.cross(e2).length() * 0.5f;
		d = Dot(n, vertices[0].p);
	}
	Vec3 centre() const
	{
		return (vertices[0].p + vertices[1].p + vertices[2].p) / 3.0f;
	}
	// Add code here
	bool rayIntersect(const Ray& r, float& t, float& u, float& v) const
	{
		const Vec3 E1 = vertices[1].p - vertices[0].p;
		const Vec3 E2 = vertices[2].p - vertices[0].p;
		const Vec3 S = r.o - vertices[0].p;
		const Vec3 S1 = Cross(r.dir, E2);
		const Vec3 S2 = Cross(S, E1);

		const float det = Dot(E1, S1);
		const float detEpsilon = 1.0e-8f;
		if (fabs(det) < detEpsilon) {
			return false;
		}

		const float invDet = 1.0f / det;
		const float beta = Dot(S, S1) * invDet;
		if (beta < 0.0f || beta > 1.0f) { return false; }

		const float gamma = Dot(r.dir, S2) * invDet;
		if (gamma < 0.0f || (beta + gamma) > 1.0f) { return false; }

		t = Dot(E2, S2) * invDet;
		if (t < 0.0f) { return false; }

		u = 1.0f - beta - gamma;
		v = beta;

		return true;
	}
	void interpolateAttributes(const float alpha, const float beta, const float gamma, Vec3& interpolatedNormal, float& interpolatedU, float& interpolatedV) const
	{
		interpolatedNormal = vertices[0].normal * alpha + vertices[1].normal * beta + vertices[2].normal * gamma;
		interpolatedNormal = interpolatedNormal.normalize();
		interpolatedU = vertices[0].u * alpha + vertices[1].u * beta + vertices[2].u * gamma;
		interpolatedV = vertices[0].v * alpha + vertices[1].v * beta + vertices[2].v * gamma;
	}
	// Add code here
	Vec3 sample(Sampler* sampler, float& pdf)
	{
		float r1 = sampler->next();
		float r2 = sampler->next();

		float sqrtR1 = sqrt(r1);

		float alpha = 1 - sqrtR1;
		float beta = r2 * sqrtR1;
		float gamma = 1 - alpha - beta;

		pdf = 1 / area;

		return vertices[0].p * alpha + vertices[1].p * beta + vertices[2].p * gamma;
	}
	Vec3 gNormal()
	{
		return (n * (Dot(vertices[0].normal, n) > 0 ? 1.0f : -1.0f));
	}
	AABB bounds() {
		AABB bounds;
		bounds.reset();
		bounds.extend(vertices[0].p);
		bounds.extend(vertices[1].p);
		bounds.extend(vertices[2].p);
		return bounds;
	}
};

class Sphere
{
public:
	Vec3 centre;
	float radius;
	void init(Vec3& _centre, float _radius)
	{
		centre = _centre;
		radius = _radius;
	}
	// Add code here
	bool rayIntersect(Ray& r, float& t)
	{
		Vec3 length = r.o - centre;
		float a = Dot(r.dir, r.dir);
		float b = 2* Dot(length, r.dir);
		float c = Dot(length, length) - radius * radius;
		float d = b * b - 4 * a * c;

		if (d < 0.0f) {
			return false;
		}

		float sqrtD = sqrt(d);

		float t1 = (-b - sqrtD) / (2.0f * a);
		float t2 = (-b + sqrtD) / (2.0f * a);
		if (t1 > EPSILON) {
			t = t1;
			return true;
		}

		if (t2 > EPSILON) {
			t = t2;
			return true;
		}
		return false;
	}
};

struct IntersectionData
{
	unsigned int ID;
	float t;
	float alpha;
	float beta;
	float gamma;
};

#define MAXNODE_TRIANGLES 8
#define TRAVERSE_COST 1.0f
#define TRIANGLE_COST 2.0f
#define BUILD_BINS 32

class BVHNode
{
private:
	int _start;
	int _count;

public:
	AABB bounds;
	BVHNode* r;
	BVHNode* l;
	// This can store an offset and number of triangles in a global triangle list for example
	// But you can store this however you want!
	// unsigned int offset;
	// unsigned char num;
	BVHNode()
	{
		r = NULL;
		l = NULL;
		_start = 0;
		_count = 0;
	}
	~BVHNode() {
		delete l;
		delete r;
	}

	bool isLeaf() {
		return l == NULL && r == NULL;
	}

	// Note there are several options for how to implement the build method. Update this as required
	void build(std::vector<Triangle>& inputTriangles)
	{
		buildRecursive(inputTriangles, 0, inputTriangles.size());
	}
	void traverse(const Ray& ray, const std::vector<Triangle>& triangles, IntersectionData& intersection)
	{
		float boxT;
		if (!bounds.rayAABB(ray, boxT)) return;
		if (boxT > intersection.t) return;

		if (isLeaf()) {
			for (int i = _start; i < _start + _count; i++) {
				float t, u, v;
				if (triangles[i].rayIntersect(ray, t, u, v)) {
					if (t > EPSILON && t < intersection.t) {
						intersection.t = t;
						intersection.ID = i;
						intersection.alpha = u;
						intersection.beta = v;
						intersection.gamma = 1.0f - u - v;
					}
				}
			}
			return;
		}

		l->traverse(ray, triangles, intersection);
		r->traverse(ray, triangles, intersection);
	}
	IntersectionData traverse(const Ray& ray, const std::vector<Triangle>& triangles)
	{
		IntersectionData intersection;
		intersection.t = FLT_MAX;
		traverse(ray, triangles, intersection);
		return intersection;
	}
	bool traverseVisible(const Ray& ray, const std::vector<Triangle>& triangles, const float maxT)
	{
		float boxT;
		if (!bounds.rayAABB(ray, boxT))
			return true;

		if (boxT > maxT)
			return true;

		if (isLeaf()) {
			for (int i = _start; i < _start + _count; i++) {
				float t, u, v;
				if (triangles[i].rayIntersect(ray, t, u, v)) {
					if (t > EPSILON && t < maxT) {
						return false;
					}
				}
			}
			return true;
		}

		if (!l->traverseVisible(ray, triangles, maxT))
			return false;

		if (!r->traverseVisible(ray, triangles, maxT))
			return false;

		return true;
	}
private:
	struct Bin {
		AABB bounds;
		int count;

		Bin() {
			bounds.reset();
			count = 0;
		}
	};
	static float getAxisStepSize(const Vec3& v, int axis) {
		if (axis == 0) return v.x;
		if (axis == 1) return v.y;
		return v.z;
	}
	static int getBinIndex(const Triangle& tri, int axis, const AABB& centroidBounds) {
		float c = getAxisStepSize(tri.centre(), axis);
		float minC = getAxisStepSize(centroidBounds.min, axis);
		float maxC = getAxisStepSize(centroidBounds.max, axis);
		float extent = maxC - minC;

		float normalized = (c - minC) / extent;
		int binIndex = static_cast<int>(normalized * BUILD_BINS);

		if (binIndex >= BUILD_BINS) binIndex = BUILD_BINS - 1;

		return binIndex;
	}
	int divideTrianglesByBin(std::vector<Triangle>& inputTriangles, int start, int count, int axis, int split, AABB& centroidBounds) {
		auto mid = std::partition(
			inputTriangles.begin() + start,
			inputTriangles.begin() + start + count,
			[&](const Triangle& tri)
			{
				int binIndex = getBinIndex(tri, axis, centroidBounds);
				return binIndex <= split;
			}
		);

		return mid - (inputTriangles.begin() + start);
	}
	void buildRecursive(std::vector<Triangle>& inputTriangles, int start, int count) {
		_start = start;
		_count = count;

		l = nullptr;
		r = nullptr;

		bounds.reset();

		for (int i = start; i < start + count; i++) {
			bounds.extend(inputTriangles[i].bounds());
		}

		if (count <= MAXNODE_TRIANGLES) {
			return;
		}

		float parentArea = bounds.area();

		if (parentArea <= EPSILON) {
			return;
		}

		AABB centroidBounds;
		centroidBounds.reset();

		for (int i = start; i < start + count; i++) {
			centroidBounds.extend(inputTriangles[i].centre());
		}

		Vec3 centroidBoundSize = centroidBounds.max - centroidBounds.min;

		if (centroidBoundSize.x <= EPSILON && centroidBoundSize.y <= EPSILON && centroidBoundSize.z <= EPSILON) {
			return;
		}

		float leafCost = TRIANGLE_COST * count;

		float bestCost = FLT_MAX;
		int bestAxis = -1;
		int bestSplit = -1;

		for (int axis = 0; axis < 3; axis++) {
			float maxValueOnAxis = getAxisStepSize(centroidBounds.max, axis);
			float minValueOnAxis = getAxisStepSize(centroidBounds.min, axis);

			float axisLength = maxValueOnAxis - minValueOnAxis;

			if (axisLength <= EPSILON) {
				continue;
			}

			Bin bins[BUILD_BINS];

			for (int i = start; i < start + count; i++) {
				int binIndex = getBinIndex(inputTriangles[i], axis, centroidBounds);
				bins[binIndex].count++;
				bins[binIndex].bounds.extend(inputTriangles[i].bounds());
			}

			AABB leftBounds[BUILD_BINS - 1];
			AABB rightBounds[BUILD_BINS - 1];

			int leftCount[BUILD_BINS - 1];
			int rightCount[BUILD_BINS - 1];

			AABB accumulatedLeftBounds;
			accumulatedLeftBounds.reset();

			int accumulatedLeftCount = 0;

			for (int i = 0; i < BUILD_BINS - 1; i++) {
				if (bins[i].count > 0) {
					accumulatedLeftBounds.extend(bins[i].bounds);
				}

				accumulatedLeftCount += bins[i].count;

				leftBounds[i] = accumulatedLeftBounds;
				leftCount[i] = accumulatedLeftCount;
			}

			AABB accumulatedRightBounds;
			accumulatedRightBounds.reset();

			int accumulatedRightCount = 0;

			for (int i = BUILD_BINS - 1; i > 0; i--) {
				if (bins[i].count > 0) {
					accumulatedRightBounds.extend(bins[i].bounds);
				}

				accumulatedRightCount += bins[i].count;

				rightBounds[i - 1] = accumulatedRightBounds;
				rightCount[i - 1] = accumulatedRightCount;
			}

			for (int split = 0; split < BUILD_BINS - 1; split++) {
				if (leftCount[split] == 0 || rightCount[split] == 0)
				{
					continue;
				}

				float leftArea = leftBounds[split].area();
				float rightArea = rightBounds[split].area();

				float leftCost = (leftArea / parentArea) * leftCount[split];
				float rightCost = (rightArea / parentArea) * rightCount[split];
				float cost = TRAVERSE_COST + TRIANGLE_COST * (leftCost + rightCost);

				if (cost < bestCost) {
					bestCost = cost;
					bestAxis = axis;
					bestSplit = split;
				}
			}
		}

		if (bestCost >= leafCost) {
			return;
		}

		int leftCountFinal = divideTrianglesByBin(inputTriangles, start, count, bestAxis, bestSplit, centroidBounds);
		int rightCountFinal = count - leftCountFinal;

		_count = 0;

		l = new BVHNode();
		r = new BVHNode();

		l->buildRecursive(inputTriangles, start, leftCountFinal);
		r->buildRecursive(inputTriangles, start + leftCountFinal, rightCountFinal);
	}
};