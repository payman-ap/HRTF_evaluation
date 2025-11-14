#include "hrtf_convolver.h"

void HRTFConvolver::loadHRIR(const std::pair<std::vector<float>, std::vector<float>>& hrir) {
    hrirL = hrir.first;
    hrirR = hrir.second;
}

std::vector<float> HRTFConvolver::convolve(const std::vector<float>& input) {
    size_t len = input.size() + hrirL.size() - 1;
    std::vector<float> outL(len, 0.0f), outR(len, 0.0f);

    for (size_t n = 0; n < input.size(); ++n) {
        for (size_t k = 0; k < hrirL.size(); ++k) {
            outL[n + k] += input[n] * hrirL[k];
            outR[n + k] += input[n] * hrirR[k];
        }
    }

    // Interleave for stereo output
    std::vector<float> interleaved(input.size() * 2);
    for (size_t i = 0; i < input.size(); ++i) {
        interleaved[2 * i]     = outL[i];
        interleaved[2 * i + 1] = outR[i];
    }
    return interleaved;
}
