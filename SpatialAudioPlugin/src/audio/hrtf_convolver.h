#pragma once
#include <vector>

class HRTFConvolver {
public:
    void loadHRIR(const std::pair<std::vector<float>, std::vector<float>>& hrir);
    std::vector<float> convolve(const std::vector<float>& monoInput);

private:
    std::vector<float> hrirL, hrirR;
};
