#include "sofa_loader.h"
#include <mysofa.h>
#include <iostream>
#include <cmath>

bool SOFALoader::load(const std::string& path) {
    int err = 0;
    sofa = mysofa_load(path.c_str(), &err);
    if (!sofa || err != MYSOFA_OK) {
        std::cerr << "Failed to load SOFA: " << mysofa_strerror(err) << "\n";
        return false;
    }
    mysofa_tocartesian(sofa);
    mysofa_resample(sofa, 44100, 0.01f);
    mysofa_neighborhood_init(sofa);
    return true;
}

std::pair<std::vector<float>, std::vector<float>> SOFALoader::getHRIR(float azimuth, float elevation) {
    int index = mysofa_find_nearest(sofa, azimuth, elevation, 1.0f);

    int nSamples = sofa->N;
    std::vector<float> left(hrirL(nSamples));
    std::vector<float> right(hrirR(nSamples));

    for (int i = 0; i < nSamples; ++i) {
        left[i] = sofa->Data.IR[index * 2 * nSamples + i];
        right[i] = sofa->Data.IR[index * 2 * nSamples + nSamples + i];
    }

    return {left, right};
}
