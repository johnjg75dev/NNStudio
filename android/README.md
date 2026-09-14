# Neural Trainer - Android Application

A high-performance neural network trainer for Android, built with Kotlin and C++ NDK.

## Project Structure

```
android/
├── app/
│   ├── src/main/
│   │   ├── java/com/neuraltrainer/
│   │   │   ├── core/           # Core ML engine (Kotlin + NDK wrapper)
│   │   │   └── ui/             # UI components
│   │   ├── cpp/                # Native C++ neural network engine
│   │   │   ├── include/core/   # C++ headers
│   │   │   └── src/            # C++ implementations
│   │   ├── res/                # Android resources
│   │   └── AndroidManifest.xml
│   └── build.gradle.kts
├── build.gradle.kts
└── settings.gradle.kts
```

## Features

- **High-Performance NDK Core**: C++ neural network engine with NEON optimizations
- **Multiple Activations**: ReLU, Leaky ReLU, Tanh, Sigmoid, GELU, Swish
- **Multiple Optimizers**: SGD, Momentum, RMSProp, Adam, AdamW
- **Multiple Loss Functions**: MSE, BCE, MAE
- **Real-time Visualization**: Network graph and loss chart
- **Training Tasks**: XOR, logic gates, 7-segment, parity, sine, circle, spiral, autoencoder

## Building

### Prerequisites
- Android Studio Arctic Fox or later
- Android SDK 26+
- NDK r25+
- CMake 3.22+

### Build Steps
1. Open the `android` folder in Android Studio
2. Sync Gradle files
3. Build → Make Project
4. Run on device or emulator

## Architecture

The app uses a hybrid architecture:
- **Kotlin**: UI, lifecycle management, coroutines
- **C++ NDK**: Core neural network computations (forward/backward pass, training)
- **JNI Bridge**: Communication between Kotlin and native code

## Performance Optimizations

- ARM NEON SIMD instructions for matrix operations
- Multi-threaded training via Kotlin coroutines
- Efficient memory management with RAII in C++
- Minimal JNI overhead with batched operations

## License

MIT License
