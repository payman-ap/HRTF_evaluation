#include "sofa/sofa_loader.h"
#include "audio/noise_generator.h"
#include "audio/hrtf_convolver.h"

#include <portaudio.h>
#include <iostream>

constexpr int SAMPLE_RATE = 44100;
constexpr int BUFFER_SIZE = 512;

struct AudioData {
    NoiseGenerator noise;
    HRTFConvolver convolver;
};

int audioCallback(const void* inputBuffer, void* outputBuffer,
                  unsigned long framesPerBuffer,
                  const PaStreamCallbackTimeInfo* timeInfo,
                  PaStreamCallbackFlags statusFlags,
                  void* userData) {
    auto* data = static_cast<AudioData*>(userData);
    float* out = static_cast<float*>(outputBuffer);

    std::vector<float> whiteNoise = data->noise.generate(framesPerBuffer);
    std::vector<float> stereoOutput = data->convolver.convolve(whiteNoise);

    for (unsigned long i = 0; i < framesPerBuffer * 2; ++i)
        out[i] = stereoOutput[i];

    return paContinue;
}

int main() {
    SOFALoader loader;
    if (!loader.load("path_to_your_file.sofa")) {
        std::cerr << "Failed to load SOFA file.\n";
        return 1;
    }

    // Get HRTFs at 0 azimuth / 0 elevation
    auto hrir = loader.getHRIR(0.0f, 0.0f);

    HRTFConvolver convolver;
    convolver.loadHRIR(hrir);

    NoiseGenerator noiseGen;

    AudioData audioData {noiseGen, convolver};

    Pa_Initialize();

    PaStream* stream;
    Pa_OpenDefaultStream(&stream, 0, 2, paFloat32,
                         SAMPLE_RATE, BUFFER_SIZE, audioCallback, &audioData);
    Pa_StartStream(stream);

    std::cout << "Playing white noise spatialized with HRTF. Press Enter to stop...\n";
    std::cin.get();

    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();

    return 0;
}
