#pragma once

#include "Core.h"
#define STB_IMAGE_IMPLEMENTATION
#include "external/stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define __STDC_LIB_EXT1__
#include "external/stb_image/stb_image_write.h"

// Stop warnings about buffer overruns if size is zero. Size should never be zero and if it is the code handles it.
#pragma warning( disable : 6386)

constexpr float texelScale = 1.0f / 255.0f;

class Texture
{
public:
	Colour* texels;
	float* alpha;
	int width;
	int height;
	int channels;
	void loadDefault()
	{
		width = 1;
		height = 1;
		channels = 3;
		texels = new Colour[1];
		texels[0] = Colour(1.0f, 1.0f, 1.0f);
	}
	void load(std::string filename)
	{
		alpha = NULL;
		if (filename.find(".hdr") != std::string::npos)
		{
			float* textureData = stbi_loadf(filename.c_str(), &width, &height, &channels, 0);
			if (width == 0 || height == 0)
			{
				loadDefault();
				return;
			}
			texels = new Colour[width * height];
			for (int i = 0; i < (width * height); i++)
			{
				texels[i] = Colour(textureData[i * channels], textureData[(i * channels) + 1], textureData[(i * channels) + 2]);
			}
			stbi_image_free(textureData);
			return;
		}
		unsigned char* textureData = stbi_load(filename.c_str(), &width, &height, &channels, 0);
		if (width == 0 || height == 0)
		{
			loadDefault();
			return;
		}
		texels = new Colour[width * height];
		for (int i = 0; i < (width * height); i++)
		{
			texels[i] = Colour(textureData[i * channels] / 255.0f, textureData[(i * channels) + 1] / 255.0f, textureData[(i * channels) + 2] / 255.0f);
		}
		if (channels == 4)
		{
			alpha = new float[width * height];
			for (int i = 0; i < (width * height); i++)
			{
				alpha[i] = textureData[(i * channels) + 3] / 255.0f;
			}
		}
		stbi_image_free(textureData);
	}
	Colour sample(const float tu, const float tv) const
	{
		Colour tex;
		float u = std::max(0.0f, fabsf(tu)) * width;
		float v = std::max(0.0f, fabsf(tv)) * height;
		int x = floorf(u);
		int y = floorf(v);
		float frac_u = u - x;
		float frac_v = v - y;
		float w0 = (1.0f - frac_u) * (1.0f - frac_v);
		float w1 = frac_u * (1.0f - frac_v);
		float w2 = (1.0f - frac_u) * frac_v;
		float w3 = frac_u * frac_v;
		x = x % width;
		y = y % height;
		Colour s[4];
		s[0] = texels[y * width + x];
		s[1] = texels[y * width + ((x + 1) % width)];
		s[2] = texels[((y + 1) % height) * width + x];
		s[3] = texels[((y + 1) % height) * width + ((x + 1) % width)];
		tex = (s[0] * w0) + (s[1] * w1) + (s[2] * w2) + (s[3] * w3);
		return tex;
	}
	float sampleAlpha(const float tu, const float tv) const
	{
		if (alpha == NULL)
		{
			return 1.0f;
		}
		float tex;
		float u = std::max(0.0f, fabsf(tu)) * width;
		float v = std::max(0.0f, fabsf(tv)) * height;
		int x = floorf(u);
		int y = floorf(v);
		float frac_u = u - x;
		float frac_v = v - y;
		float w0 = (1.0f - frac_u) * (1.0f - frac_v);
		float w1 = frac_u * (1.0f - frac_v);
		float w2 = (1.0f - frac_u) * frac_v;
		float w3 = frac_u * frac_v;
		x = x % width;
		y = y % height;
		float s[4];
		s[0] = alpha[y * width + x];
		s[1] = alpha[y * width + ((x + 1) % width)];
		s[2] = alpha[((y + 1) % height) * width + x];
		s[3] = alpha[((y + 1) % height) * width + ((x + 1) % width)];
		tex = (s[0] * w0) + (s[1] * w1) + (s[2] * w2) + (s[3] * w3);
		return tex;
	}
	~Texture()
	{
		delete[] texels;
		if (alpha != NULL)
		{
			delete alpha;
		}
	}
};

class ImageFilter
{
public:
	virtual float filter(const float x, const float y) const = 0;
	virtual int size() const = 0;
};

class BoxFilter : public ImageFilter
{
public:
	float filter(float x, float y) const
	{
		if (fabsf(x) < 0.5f && fabs(y) < 0.5f)
		{
			return 1.0f;
		}
		return 0;
	}
	int size() const
	{
		return 1;
	}
};

class GaussianFilter : public ImageFilter
{
public:
	float radius;
	float alpha;

	GaussianFilter(float r = 2.0f, float a = 2.0f)
		: radius(r), alpha(a) {
	}

	float gaussian(float d) const {
		if (fabsf(d) >= radius) {
			return 0.0f;
		}
		float value = expf(-alpha * d * d) - expf(-alpha * radius * radius);
		return std::max(0.0f, value);
	}

	float filter(const float x, const float y) const {
		return gaussian(x) * gaussian(y);
	}

	int size() const
	{
		return ceil(radius);
	}
};

class Film
{
public:
	Colour* film;
	unsigned int width;
	unsigned int height;
	int SPP;
	ImageFilter* filter;

	void splat(const float x, const float y, const Colour& L) {
		// Code to splat a smaple with colour L into the image plane using an ImageFilter
		float filterWeights[25];
		unsigned int indices[25];
		unsigned int rows[25];
		unsigned int used = 0;
		unsigned int rowCount = 0;
		float total = 0;
		int size = filter->size();
		for (int i = -size; i <= size; i++) {
			for (int j = -size; j <= size; j++) {
				int px = x + j;
				int py = y + i;
				if (px >= 0 && px < width && py >= 0 && py < height) {
					indices[used] = (py * width) + px;
					filterWeights[used] = filter->filter(px - x, py - y);
					total += filterWeights[used];
					bool rowFound = false;
					for (unsigned int r = 0; r < rowCount; r++) {
						if (rows[r] == py) {
							rowFound = true;
							break;
						}
					}
					if (!rowFound) {
						rows[rowCount] = py;
						rowCount++;
					}
					used++;
				}
			}
		}
		if (total <= 0.0f) return;

		for (unsigned int i = 0; i < used; i++) {
			film[indices[i]] = film[indices[i]] + (L * filterWeights[i] / total);
		}
	}
	void splatToTile(float x, float y, Colour& L, Colour* tile, int tileXStart, int tileYStart, unsigned int tileWidth, unsigned int tileHeight) {
		float total = 0;
		int size = filter->size();
		for (int i = -size; i <= size; i++) {
			for (int j = -size; j <= size; j++) {
				int px = x + j;
				int py = y + i;
				int tx = px - tileXStart;
				int ty = py - tileYStart;
				if (px >= 0 && px < width && py >= 0 && py < height &&
					tx >= 0 && tx < tileWidth && ty >= 0 && ty < tileHeight) {
					total += filter->filter(px - x, py - y);
				}
			}
		}
		if (total <= 0.0f) return;

		for (int i = -size; i <= size; i++) {
			for (int j = -size; j <= size; j++) {
				int px = x + j;
				int py = y + i;
				int tx = px - tileXStart;
				int ty = py - tileYStart;
				if (px >= 0 && px < width && py >= 0 && py < height &&
					tx >= 0 && tx < tileWidth && ty >= 0 && ty < tileHeight) {
					int index = ty * tileWidth + tx;
					tile[index] = tile[index] + (L * filter->filter(px - x, py - y) / total);
				}
			}
		}
	}
	void mergeTile(Colour* tile, int tileXStart, int tileYStart, unsigned int tileWidth, unsigned int tileHeight) {
		for (int ty = 0; ty < tileHeight; ty++) {
			int y = tileYStart + ty;
			if (y < 0 || y >= height) continue;

			for (int tx = 0; tx < tileWidth; tx++) {
				int x = tileXStart + tx;
				if (x < 0 || x >= width) continue;

				int filmIndex = y * width + x;
				int tileIndex = ty * tileWidth + tx;
				film[filmIndex] = film[filmIndex] + tile[tileIndex];
			}
		}
	}
	void tonemap(int x, int y, unsigned char& r, unsigned char& g, unsigned char& b, float exposure = 1.0f)
	{
		// Return a tonemapped pixel at coordinates x, y
		Colour colour = film[y * width + x] / SPP;
		colour = colour * exposure;

		// Reinhard Global
		colour.r = colour.r / (1.0f + colour.r);
		colour.g = colour.g / (1.0f + colour.g);
		colour.b = colour.b / (1.0f + colour.b);

		colour.r = powf(colour.r, 1.0f / 2.2f);
		colour.g = powf(colour.g, 1.0f / 2.2f);
		colour.b = powf(colour.b, 1.0f / 2.2f);

		r = (unsigned char)(colour.r * 255.0f);
		g = (unsigned char)(colour.g * 255.0f);
		b = (unsigned char)(colour.b * 255.0f);
	}
	// Do not change any code below this line
	void init(int _width, int _height, ImageFilter* _filter)
	{
		width = _width;
		height = _height;
		film = new Colour[width * height];
		clear();
		filter = _filter;
	}
	void clear()
	{
		memset(film, 0, width * height * sizeof(Colour));
		SPP = 0;
	}
	void incrementSPP()
	{
		SPP++;
	}
	void save(std::string filename)
	{
		Colour* hdrpixels = new Colour[width * height];
		for (unsigned int i = 0; i < (width * height); i++)
		{
			hdrpixels[i] = film[i] / (float)SPP;
		}
		stbi_write_hdr(filename.c_str(), width, height, 3, (float*)hdrpixels);
		delete[] hdrpixels;
	}
};