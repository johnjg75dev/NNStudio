# Neural Network Trainer - Android NDK Core
# High-performance neural network engine in C++

cmake_minimum_required(VERSION 3.22.1)
project("neuraltrainer_core")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable NEON optimizations for ARM
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon -mfloat-abi=softfp")
endif()

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -ffast-math -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")

# Source files
set(SOURCES
    src/core/activations.cpp
    src/core/losses.cpp
    src/core/layers/dense_layer.cpp
    src/core/layers/conv_layer.cpp
    src/core/layers/rnn_layer.cpp
    src/core/layers/transformer_layer.cpp
    src/core/optimizer.cpp
    src/core/network.cpp
    src/jni/native_interface.cpp
)

# Header files
set(HEADERS
    include/core/activations.h
    include/core/losses.h
    include/core/layers/base_layer.h
    include/core/layers/dense_layer.h
    include/core/layers/conv_layer.h
    include/core/layers/rnn_layer.h
    include/core/layers/transformer_layer.h
    include/core/optimizer.h
    include/core/network.h
    include/jni/native_interface.h
)

# Create shared library
add_library(neuraltrainer_core SHARED ${SOURCES})

# Include directories
target_include_directories(neuraltrainer_core PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

# Link against Android log library
find_library(log-lib log)
target_link_libraries(neuraltrainer_core ${log-lib})
