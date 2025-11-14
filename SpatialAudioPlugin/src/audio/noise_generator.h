#pragma once
#include <vector>
#include <random>

class NoiseGenerator {
public:
    NoiseGenerator();

    std::vector<float> generate(size_t samples);

private:
    std::mt19937 rng;
    std::uniform_real_distribution<float> dist;
};
