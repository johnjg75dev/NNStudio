#include "jni/native_interface.h"
#include <android/log.h>
#include <sstream>
#include <algorithm>
#include <cctype>

#define LOG_TAG "NeuralTrainer"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

using namespace neuraltrainer;
using namespace neuraltrainer::core;
using namespace neuraltrainer::jni;

// Simple JSON parsing helpers
static std::string jstringToString(JNIEnv* env, jstring str) {
    if (!str) return "";
    const char* chars = env->GetStringUTFChars(str, nullptr);
    std::string result(chars);
    env->ReleaseStringUTFChars(str, chars);
    return result;
}

static jstring stringToJstring(JNIEnv* env, const std::string& str) {
    return env->NewStringUTF(str.c_str());
}

// Trim whitespace
static std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\n\r");
    return s.substr(start, end - start + 1);
}

// Parse JSON array of doubles
static std::vector<double> parseDoubleArray(const std::string& json) {
    std::vector<double> result;
    std::string trimmed = trim(json);
    
    // Remove brackets
    if (trimmed.front() == '[' && trimmed.back() == ']') {
        trimmed = trimmed.substr(1, trimmed.size() - 2);
    }
    
    std::stringstream ss(trimmed);
    std::string item;
    
    while (std::getline(ss, item, ',')) {
        try {
            double val = std::stod(trim(item));
            result.push_back(val);
        } catch (...) {
            // Skip invalid entries
        }
    }
    
    return result;
}

// Parse dataset JSON: [{"x":[...],"y":[...]}, ...]
static std::vector<std::pair<std::vector<double>, std::vector<double>>> parseDataset(
    const std::string& json) {
    
    std::vector<std::pair<std::vector<double>, std::vector<double>>> dataset;
    
    // Find all {"x":..., "y":...} objects
    size_t pos = 0;
    while ((pos = json.find('{', pos)) != std::string::npos) {
        size_t end = json.find('}', pos);
        if (end == std::string::npos) break;
        
        std::string obj = json.substr(pos, end - pos + 1);
        
        // Extract x array
        size_t xPos = obj.find("\"x\"");
        size_t yPos = obj.find("\"y\"");
        
        if (xPos != std::string::npos && yPos != std::string::npos) {
            size_t xStart = obj.find('[', xPos);
            size_t xEnd = obj.find(']', xStart);
            size_t yStart = obj.find('[', yPos);
            size_t yEnd = obj.find(']', yStart);
            
            if (xStart != std::string::npos && xEnd != std::string::npos &&
                yStart != std::string::npos && yEnd != std::string::npos) {
                
                std::vector<double> x = parseDoubleArray(obj.substr(xStart, xEnd - xStart + 1));
                std::vector<double> y = parseDoubleArray(obj.substr(yStart, yEnd - yStart + 1));
                
                if (!x.empty() && !y.empty()) {
                    dataset.push_back({x, y});
                }
            }
        }
        
        pos = end + 1;
    }
    
    return dataset;
}

// Convert vector to JSON array string
static std::string vectorToJson(const std::vector<double>& vec) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i > 0) oss << ",";
        oss << std::fixed << std::setprecision(6) << vec[i];
    }
    oss << "]";
    return oss.str();
}

extern "C" {

JNIEXPORT jint JNICALL Java_com_neuraltrainer_core_NativeLib_initialize(JNIEnv* env, jobject thiz) {
    LOGD("Initializing NeuralTrainer native library");
    
    // Clean up any existing sessions
    getSessionManager().clearAll();
    
    return 0; // Success
}

JNIEXPORT jint JNICALL Java_com_neuraltrainer_core_NativeLib_buildNetwork(
    JNIEnv* env, jobject thiz,
    jint inputs, jint outputs,
    jstring hiddenLayers,
    jstring activation,
    jstring optimizer,
    jdouble lr,
    jstring loss) {
    
    try {
        std::string actName = jstringToString(env, activation);
        std::string optName = jstringToString(env, optimizer);
        std::string lossName = jstringToString(env, loss);
        
        // Get activation function
        const Activation* act = ActivationRegistry::get(actName);
        if (!act) {
            LOGE("Unknown activation: %s", actName.c_str());
            return -1;
        }
        
        // Get loss function
        const LossFunction* lossFn = LossRegistry::get(lossName);
        if (!lossFn) {
            LOGE("Unknown loss: %s", lossName.c_str());
            return -1;
        }
        
        // Create optimizer
        auto opt = OptimizerFactory::create(optName, lr, 0.9, 0.0);
        
        // Build layer list
        std::vector<std::unique_ptr<Layer>> layers;
        
        // For now, create a simple MLP based on hidden layer count
        // In a full implementation, we'd parse the JSON for complex architectures
        int currSize = inputs;
        
        // Parse hidden layers from JSON (simple format: "[64,32]" means two hidden layers)
        std::string layersJson = jstringToString(env, hiddenLayers);
        if (!layersJson.empty() && layersJson != "[]") {
            // Remove brackets and split by comma
            layersJson = layersJson.substr(1, layersJson.size() - 2);
            std::stringstream ss(layersJson);
            std::string item;
            
            while (std::getline(ss, item, ',')) {
                int neurons = std::stoi(item);
                layers.push_back(std::make_unique<DenseLayer>(currSize, neurons, act, false));
                currSize = neurons;
            }
        } else {
            // Default: single hidden layer with 16 neurons
            layers.push_back(std::make_unique<DenseLayer>(inputs, 16, act, false));
            currSize = 16;
        }
        
        // Add output layer
        layers.push_back(std::make_unique<DenseLayer>(currSize, outputs, nullptr, true));
        
        // Create network
        auto network = std::make_unique<NeuralNetwork>(std::move(layers), std::move(opt), lossFn);
        
        // Register session and return ID
        int sessionId = getSessionManager().createSession(std::move(network));
        LOGD("Created network session %d", sessionId);
        
        return sessionId;
        
    } catch (const std::exception& e) {
        LOGE("Error building network: %s", e.what());
        return -1;
    }
}

JNIEXPORT jdouble JNICALL Java_com_neuraltrainer_core_NativeLib_trainSteps(
    JNIEnv* env, jobject thiz,
    jint sessionId,
    jstring datasetJson,
    jint steps,
    jdouble lr) {
    
    auto* network = getSessionManager().getSession(sessionId);
    if (!network) {
        LOGE("Invalid session ID: %d", sessionId);
        return -1.0;
    }
    
    std::string jsonStr = jstringToString(env, datasetJson);
    
    // Parse dataset
    auto dataset = parseDataset(jsonStr);
    if (dataset.empty()) {
        LOGE("Failed to parse dataset or empty dataset");
        return -1.0;
    }
    
    double totalLoss = 0.0;
    int samplesTrained = 0;
    
    // Train for specified number of steps
    for (int step = 0; step < steps && samplesTrained < static_cast<int>(dataset.size()); ++step) {
        // Use modulo to cycle through dataset if steps > dataset size
        const auto& sample = dataset[samplesTrained % dataset.size()];
        totalLoss += network->trainStep(sample.first, sample.second);
        samplesTrained++;
    }
    
    double avgLoss = (samplesTrained > 0) ? totalLoss / samplesTrained : 0.0;
    LOGD("Trained %d steps, avg loss: %.6f", steps, avgLoss);
    
    return avgLoss;
}

JNIEXPORT jstring JNICALL Java_com_neuraltrainer_core_NativeLib_predict(
    JNIEnv* env, jobject thiz,
    jint sessionId,
    jstring inputJson) {
    
    auto* network = getSessionManager().getSession(sessionId);
    if (!network) {
        LOGE("Invalid session ID: %d", sessionId);
        return stringToJstring(env, "{\"error\":\"Invalid session\"}");
    }
    
    std::string jsonStr = jstringToString(env, inputJson);
    std::vector<double> input = parseDoubleArray(jsonStr);
    
    if (input.empty()) {
        return stringToJstring(env, "{\"error\":\"Invalid input\"}");
    }
    
    // Run prediction
    std::vector<double> output = network->predict(input, false);
    std::string outputJson = vectorToJson(output);
    
    return stringToJstring(env, outputJson.c_str());
}

JNIEXPORT jstring JNICALL Java_com_neuraltrainer_core_NativeLib_getSnapshot(
    JNIEnv* env, jobject thiz,
    jint sessionId) {
    
    auto* network = getSessionManager().getSession(sessionId);
    if (!network) {
        return stringToJstring(env, "{\"error\":\"Invalid session\"}");
    }
    
    return stringToJstring(env, network->toJson());
}

JNIEXPORT jstring JNICALL Java_com_neuraltrainer_core_NativeLib_exportNetwork(
    JNIEnv* env, jobject thiz,
    jint sessionId) {
    
    return Java_com_neuraltrainer_core_NativeLib_getSnapshot(env, thiz, sessionId);
}

JNIEXPORT jint JNICALL Java_com_neuraltrainer_core_NativeLib_importNetwork(
    JNIEnv* env, jobject thiz,
    jstring json) {
    
    // TODO: Implement network import from JSON
    // This would parse the JSON and reconstruct the network
    
    std::string jsonStr = jstringToString(env, json);
    LOGD("Import network called with JSON length: %zu", jsonStr.size());
    
    // Placeholder - returns -1 (not implemented)
    return -1;
}

JNIEXPORT void JNICALL Java_com_neuraltrainer_core_NativeLib_destroySession(
    JNIEnv* env, jobject thiz,
    jint sessionId) {
    
    getSessionManager().destroySession(sessionId);
    LOGD("Destroyed session %d", sessionId);
}

JNIEXPORT void JNICALL Java_com_neuraltrainer_core_NativeLib_cleanup(JNIEnv* env, jobject thiz) {
    getSessionManager().clearAll();
    LOGD("Cleaned up all sessions");
}

} // extern "C"
