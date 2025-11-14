#include "noise_generator.h"

NoiseGenerator::NoiseGenerator()
    : rng(std::random_device{}()), dist(-1.0f, 1.0f) {}

std::vector<float> NoiseGenerator::generate(size_t samples) {
    std::vector<float> buffer(samples);
    for (auto& sample : buffer)
        sample = dist(rng);
    return buffer;
}
