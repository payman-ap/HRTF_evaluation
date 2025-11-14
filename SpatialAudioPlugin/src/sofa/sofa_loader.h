#pragma once
#include <vector>
#include <string>

class SOFALoader {
public:
    bool load(const std::string& path);

    // Returns {left HRIR, right HRIR}
    std::pair<std::vector<float>, std::vector<float>> getHRIR(float azimuth, float elevation);

private:
    struct MYSOFA_HRTF* sofa = nullptr;
};
