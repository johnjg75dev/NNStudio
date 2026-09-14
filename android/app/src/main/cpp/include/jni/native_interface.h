#ifndef NEURALTRAINER_NATIVE_INTERFACE_H
#define NEURALTRAINER_NATIVE_INTERFACE_H

#include <jni.h>
#include <string>
#include <memory>
#include <unordered_map>
#include "core/network.h"
#include "core/optimizer.h"
#include "core/losses.h"
#include "core/layers/dense_layer.h"

namespace neuraltrainer {
namespace jni {

/**
 * Session manager - keeps track of active networks
 */
class SessionManager {
private:
    std::unordered_map<int, std::unique_ptr<core::NeuralNetwork>> sessions;
    int nextSessionId;
    
public:
    SessionManager() : nextSessionId(1) {}
    
    int createSession(std::unique_ptr<core::NeuralNetwork> network) {
        int id = nextSessionId++;
        sessions[id] = std::move(network);
        return id;
    }
    
    core::NeuralNetwork* getSession(int sessionId) {
        auto it = sessions.find(sessionId);
        return it != sessions.end() ? it->second.get() : nullptr;
    }
    
    bool destroySession(int sessionId) {
        return sessions.erase(sessionId) > 0;
    }
    
    void clearAll() {
        sessions.clear();
    }
};

// Global session manager (singleton pattern for JNI)
inline SessionManager& getSessionManager() {
    static SessionManager manager;
    return manager;
}

} // namespace jni
} // namespace neuraltrainer

// JNI Function declarations
extern "C" {

/**
 * Initialize the neural trainer native library
 */
JNIEXPORT jint JNICALL Java_com_neuraltrainer_core_NativeLib_initialize(JNIEnv* env, jobject thiz);

/**
 * Build a new neural network
 * @param inputs Number of input features
 * @param outputs Number of output neurons
 * @param hiddenLayers JSON array of layer configurations
 * @param activation Activation function name
 * @param optimizer Optimizer name
 * @param lr Learning rate
 * @param loss Loss function name
 * @return Session ID or -1 on error
 */
JNIEXPORT jint JNICALL Java_com_neuraltrainer_core_NativeLib_buildNetwork(
    JNIEnv* env, jobject thiz,
    jint inputs, jint outputs,
    jstring hiddenLayers,
    jstring activation,
    jstring optimizer,
    jdouble lr,
    jstring loss);

/**
 * Train for N steps
 * @param sessionId Session ID
 * @param datasetJson JSON-encoded dataset
 * @param steps Number of training steps
 * @param lr Learning rate
 * @return Average loss
 */
JNIEXPORT jdouble JNICALL Java_com_neuraltrainer_core_NativeLib_trainSteps(
    JNIEnv* env, jobject thiz,
    jint sessionId,
    jstring datasetJson,
    jint steps,
    jdouble lr);

/**
 * Run prediction
 * @param sessionId Session ID
 * @param inputJson JSON-encoded input vector
 * @return JSON-encoded output vector
 */
JNIEXPORT jstring JNICALL Java_com_neuraltrainer_core_NativeLib_predict(
    JNIEnv* env, jobject thiz,
    jint sessionId,
    jstring inputJson);

/**
 * Get network snapshot for visualization
 * @param sessionId Session ID
 * @return JSON-encoded snapshot
 */
JNIEXPORT jstring JNICALL Java_com_neuraltrainer_core_NativeLib_getSnapshot(
    JNIEnv* env, jobject thiz,
    jint sessionId);

/**
 * Export network to JSON
 * @param sessionId Session ID
 * @return JSON-encoded network
 */
JNIEXPORT jstring JNICALL Java_com_neuraltrainer_core_NativeLib_exportNetwork(
    JNIEnv* env, jobject thiz,
    jint sessionId);

/**
 * Import network from JSON
 * @param json JSON-encoded network
 * @return Session ID or -1 on error
 */
JNIEXPORT jint JNICALL Java_com_neuraltrainer_core_NativeLib_importNetwork(
    JNIEnv* env, jobject thiz,
    jstring json);

/**
 * Destroy a session
 * @param sessionId Session ID
 */
JNIEXPORT void JNICALL Java_com_neuraltrainer_core_NativeLib_destroySession(
    JNIEnv* env, jobject thiz,
    jint sessionId);

/**
 * Cleanup all sessions
 */
JNIEXPORT void JNICALL Java_com_neuraltrainer_core_NativeLib_cleanup(JNIEnv* env, jobject thiz);

} // extern "C"

#endif // NEURALTRAINER_NATIVE_INTERFACE_H
