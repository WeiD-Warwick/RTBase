#pragma once
#define EPSILON 1e-6f

#include "Core.h"
#include "Sampling.h"

class Ray {
public:
	Vec3 o;
	Vec3 dir;
	Vec3 invDir;
	Ray() {}
	Ray(Vec3 _o, Vec3 _d) {
		init(_o, _d);
	}

	void init(Vec3 _o, Vec3 _d) {
		o = _o;
		dir = _d;
		invDir = Vec3(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);
	}

	Vec3 at(const float t) const {
		return (o + (dir * t));
	}
};

class Plane {
public:
	Vec3 n;
	float d;
	void init(Vec3& _n, float _d) {
		n = _n;
		d = _d;
	}

	// Add code here
	bool rayIntersect(Ray& r, float& t) {
		float denom = Dot(n, r.dir);
		if (denom == 0) { return false; }

		t = (-d - Dot(n, r.o)) / denom;
		if (t < 0) return false;
		return true;
	}
};

class Triangle {
public:
	Vertex vertices[3];
	Vec3 e1; // Edge 1
	Vec3 e2; // Edge 2
	Vec3 n; // Geometric Normal
	float area; // Triangle area
	float d; // For ray triangle if needed
	unsigned int materialIndex;

	void init(Vertex v0, Vertex v1, Vertex v2, unsigned int _materialIndex) {
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

	Vec3 centre() const {
		return (vertices[0].p + vertices[1].p + vertices[2].p) / 3.0f;
	}

	// Add code here
	bool rayIntersect(const Ray& r, float& t, float& u, float& v) const {
		float denom = Dot(n, r.dir);
		if (denom == 0) { return false; }

		t = (d - Dot(n, r.o)) / denom;
		if (t < 0) { return false; }

		Vec3 p = r.at(t);
		float invArea = 1.0f / Dot(e1.cross(e2), n);
		u = Dot(e1.cross(p - vertices[1].p), n) * invArea;
		if (u < 0 || u > 1.0f) { return false; }
		v = Dot(e2.cross(p - vertices[2].p), n) * invArea;
		if (v < 0 || (u + v) > 1.0f) { return false; }

		return true;

		// to do
		// Moller-Trumbore


	}

	void interpolateAttributes(const float alpha, const float beta, const float gamma, Vec3& interpolatedNormal, float& interpolatedU, float& interpolatedV) const
	{
		interpolatedNormal = vertices[0].normal * alpha + vertices[1].normal * beta + vertices[2].normal * gamma;
		interpolatedNormal = interpolatedNormal.normalize();
		interpolatedU = vertices[0].u * alpha + vertices[1].u * beta + vertices[2].u * gamma;
		interpolatedV = vertices[0].v * alpha + vertices[1].v * beta + vertices[2].v * gamma;
	}


	Vec3 sample(Sampler* sampler, float& pdf) {

		// MonteCarlo 137

		float r1 = sampler->next();
		float r2 = sampler->next();

		float sqrtR1 = sqrt(r1);

		float alpha = 1 - sqrtR1;
		float beta = r2 * sqrtR1;
		float gamma = 1 - alpha - beta;

		pdf = 1 / area;

		return vertices[0].p * alpha + vertices[1].p * beta + vertices[2].p * gamma;
	}

	Vec3 gNormal() {
		return (n * (Dot(vertices[0].normal, n) > 0 ? 1.0f : -1.0f));
	}
};

class AABB {
public:
	Vec3 max;
	Vec3 min;
	AABB() {
		reset();
	}

	void reset() {
		max = Vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
		min = Vec3(FLT_MAX, FLT_MAX, FLT_MAX);
	}

	void extend(const Vec3 p) {
		max = Max(max, p);
		min = Min(min, p);
	}

	// Add code here
	bool rayAABB(const Ray& r, float& t) {
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

	// Add code here
	bool rayAABB(const Ray& r) {
		Vec3 Tmin = (min - r.o) * r.invDir;
		Vec3 Tmax = (max - r.o) * r.invDir;
		Vec3 Tentry = Min(Tmin, Tmax);
		Vec3 Texit = Max(Tmin, Tmax);

		float tentry = std::max(Tentry.x, std::max(Tentry.y, Tentry.z));
		float texit = std::min(Texit.x, std::min(Texit.y, Texit.z));

		return (tentry <= texit && texit >= 0.0f);
	}

	// Add code here
	float area() {
		Vec3 size = max - min;
		return ((size.x * size.y) + (size.y * size.z) + (size.x * size.z)) * 2.0f;
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

	bool rayIntersect(Ray& r, float& t)
	{
		Vec3 oc = r.o - centre;
		float a = Dot(r.dir, r.dir);
		float b = 2.0f * Dot(oc, r.dir);
		float c = Dot(oc, oc) - radius * radius;
		float disc = b * b - 4.0f * a * c;
		if (disc < 0.0f) {
			return false;
		}

		float s = sqrtf(disc);
		float inv2a = 0.5f / a;
		float t0 = (-b - s) * inv2a;
		float t1 = (-b + s) * inv2a;
		if (t0 > EPSILON) {
			t = t0;
			return true;
		}
		if (t1 > EPSILON) {
			t = t1;
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

class BVHNode {
private:
	int _start;
	int _count;

public:
	AABB bounds;
	BVHNode* r;
	BVHNode* l;

	BVHNode() {
		r = NULL;
		l = NULL;
		_start = 0;
		_count = 0;
	}

	~BVHNode() {
		if (l) delete l;
		if (r) delete r;
	}

	bool isLeaf() const {
		return l == NULL && r == NULL;
	}

	void build(std::vector<Triangle>& inputTriangles) {
		buildRecursive(inputTriangles, 0, inputTriangles.size());
	}

	void traverse(const Ray& ray, const std::vector<Triangle>& triangles, IntersectionData& intersection) {
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
						intersection.gamma = 1.0f - (u + v);
					}
				}
			}
			return;
		}

		float leftT = FLT_MAX;
		float rightT = FLT_MAX;
		bool hitLeft = l && l->bounds.rayAABB(ray, leftT) && leftT <= intersection.t;
		bool hitRight = r && r->bounds.rayAABB(ray, rightT) && rightT <= intersection.t;

		if (hitLeft && hitRight) {
			if (leftT < rightT) {
				l->traverse(ray, triangles, intersection);
				if (rightT <= intersection.t) r->traverse(ray, triangles, intersection);
			}
			else {
				r->traverse(ray, triangles, intersection);
				if (leftT <= intersection.t) l->traverse(ray, triangles, intersection);
			}
		}
		else if (hitLeft) {
			l->traverse(ray, triangles, intersection);
		}
		else if (hitRight) {
			r->traverse(ray, triangles, intersection);
		}
	}

	IntersectionData traverse(const Ray& ray, const std::vector<Triangle>& triangles) {
		IntersectionData intersection;
		intersection.t = FLT_MAX;
		traverse(ray, triangles, intersection);
		return intersection;
	}

	bool traverseVisible(const Ray& ray, const std::vector<Triangle>& triangles, const float maxT) {
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

		float leftT = FLT_MAX;
		float rightT = FLT_MAX;
		bool hitLeft = l && l->bounds.rayAABB(ray, leftT) && leftT <= maxT;
		bool hitRight = r && r->bounds.rayAABB(ray, rightT) && rightT <= maxT;

		if (hitLeft && hitRight) {
			if (leftT < rightT) {
				if (!l->traverseVisible(ray, triangles, maxT)) return false;
				if (!r->traverseVisible(ray, triangles, maxT)) return false;
			}
			else {
				if (!r->traverseVisible(ray, triangles, maxT)) return false;
				if (!l->traverseVisible(ray, triangles, maxT)) return false;
			}
		}
		else if (hitLeft) {
			if (!l->traverseVisible(ray, triangles, maxT)) return false;
		}
		else if (hitRight) {
			if (!r->traverseVisible(ray, triangles, maxT)) return false;
		}

		return true;
	}

private:
	void buildRecursive(std::vector<Triangle>& inputTriangles, int start, int count) {
		_start = start;
		_count = count;
		bounds.reset();

		for (int i = start; i < start + count; i++) {
			bounds.extend(inputTriangles[i].vertices[0].p);
			bounds.extend(inputTriangles[i].vertices[1].p);
			bounds.extend(inputTriangles[i].vertices[2].p);
		}

		if (count <= MAXNODE_TRIANGLES) {
			return;
		}

		Vec3 size = bounds.max - bounds.min;
		int axis = 0;

		if (size.y > size.x && size.y >= size.z) {
			axis = 1;
		}
		else if (size.z > size.x && size.z > size.y) {
			axis = 2;
		}

		auto axisValue = [axis](const Triangle& t) -> float {
			Vec3 centre = t.centre();
			return axis == 0 ? centre.x : (axis == 1 ? centre.y : centre.z);
		};

		std::sort(inputTriangles.begin() + start, inputTriangles.begin() + start + count, 
			[&](const Triangle& a, const Triangle& b) {
				return axisValue(a) < axisValue(b);
			}
		);

		int leftCount = count / 2;
		int rightCount = count - leftCount;

		if (leftCount == 0 || rightCount == 0) {
			return;
		}

		l = new BVHNode();
		r = new BVHNode();

		l->buildRecursive(inputTriangles, _start, leftCount);
		r->buildRecursive(inputTriangles, _start + leftCount, rightCount);
	}
};
