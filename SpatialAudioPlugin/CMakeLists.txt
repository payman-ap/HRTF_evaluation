cmake_minimum_required(VERSION 3.10)
project(SpatialAudio)

set(CMAKE_CXX_STANDARD 17)

find_package(PortAudio REQUIRED)

include_directories(
    ${PORTAUDIO_INCLUDE_DIRS}
    /usr/local/include
)

link_directories(/usr/local/lib)

add_executable(SpatialAudio
    main.cpp
    audio/noise_generator.cpp
    audio/hrtf_convolver.cpp
    sofa/sofa_loader.cpp
)

target_link_libraries(SpatialAudio
    portaudio
    mysofa
)
