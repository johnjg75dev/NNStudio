// RNN Layer header - Basic recurrent neural network layer
#ifndef NEURALTRAINER_RNN_LAYER_H
#define NEURALTRAINER_RNN_LAYER_H

#include "base_layer.h"
#include "../activations.h"
#include <vector>
#include <string>
#include <random>
#include <cmath>

namespace neuraltrainer {
namespace core {

/**
 * Simple Recurrent Neural Network (RNN) Layer
 * Supports vanilla RNN with configurable hidden size
 */
class RNNLayer : public Layer {
private:
    size_t inputSize;
    size_t hiddenSize;
    
    // Weight matrices
    std::vector<std::vector<double>> Wxh;  // Input to hidden
    std::vector<std::vector<double>> Whh;  // Hidden to hidden (recurrent)
    std::vector<std::vector<double>> Why;  // Hidden to output
    std::vector<double> bh, by;            // Biases
    
    // Gradients
    std::vector<std::vector<double>> dWxh, dWhh, dWhy;
    std::vector<double> dbh, dby;
    
    const Activation* activation;
    
    // Hidden state
    std::vector<double> hiddenState;
    
    static std::mt19937 rng;
    
public:
    /**
     * Constructor for RNN layer
     * @param inSize Input feature size
     * @param hidSize Hidden state size
     * @param act Activation function (default: tanh)
     */
    RNNLayer(size_t inSize, size_t hidSize, const Activation* act = nullptr)
        : inputSize(inSize), hiddenSize(hidSize), activation(act) {
        
        if (!activation) {
            activation = ActivationRegistry::get("tanh");
        }
        
        // Initialize weights with Xavier initialization
        double scaleXh = std::sqrt(2.0 / (inputSize + hiddenSize));
        double scaleHh = std::sqrt(2.0 / (hiddenSize + hiddenSize));
        double scaleHy = std::sqrt(2.0 / (hiddenSize + hiddenSize));
        
        std::uniform_real_distribution<double> distXh(-scaleXh, scaleXh);
        std::uniform_real_distribution<double> distHh(-scaleHh, scaleHh);
        std::uniform_real_distribution<double> distHy(-scaleHy, scaleHy);
        
        // Resize matrices
        Wxh.resize(hiddenSize, std::vector<double>(inputSize));
        Whh.resize(hiddenSize, std::vector<double>(hiddenSize));
        Why.resize(hiddenSize, std::vector<double>(hiddenSize));
        bh.resize(hiddenSize, 0.0);
        by.resize(hiddenSize, 0.0);
        
        dWxh.resize(hiddenSize, std::vector<double>(inputSize, 0.0));
        dWhh.resize(hiddenSize, std::vector<double>(hiddenSize, 0.0));
        dWhy.resize(hiddenSize, std::vector<double>(hiddenSize, 0.0));
        dbh.resize(hiddenSize, 0.0);
        dby.resize(hiddenSize, 0.0);
        
        // Initialize weights
        for (size_t i = 0; i < hiddenSize; ++i) {
            for (size_t j = 0; j < inputSize; ++j) {
                Wxh[i][j] = distXh(rng);
            }
            for (size_t j = 0; j < hiddenSize; ++j) {
                Whh[i][j] = distHh(rng);
                Why[i][j] = distHy(rng);
            }
        }
        
        // Initialize hidden state
        hiddenState.resize(hiddenSize, 0.0);
    }
    
    void resetHiddenState() {
        std::fill(hiddenState.begin(), hiddenState.end(), 0.0);
    }
    
    std::vector<double> forward(const std::vector<double>& input, bool training = true) override {
        cachedInput = input;
        
        // h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
        std::vector<double> newHidden(hiddenSize, 0.0);
        
        for (size_t i = 0; i < hiddenSize; ++i) {
            double sum = bh[i];
            
            // Input contribution
            for (size_t j = 0; j < inputSize; ++j) {
                sum += Wxh[i][j] * input[j];
            }
            
            // Recurrent contribution
            for (size_t j = 0; j < hiddenSize; ++j) {
                sum += Whh[i][j] * hiddenState[j];
            }
            
            newHidden[i] = activation->forward(sum);
        }
        
        hiddenState = newHidden;
        
        // Output: y = W_hy * h + b_y
        cachedOutput.resize(hiddenSize);
        for (size_t i = 0; i < hiddenSize; ++i) {
            double sum = by[i];
            for (size_t j = 0; j < hiddenSize; ++j) {
                sum += Why[i][j] * hiddenState[j];
            }
            cachedOutput[i] = sum;
        }
        
        return cachedOutput;
    }
    
    std::vector<double> backward(const std::vector<double>& gradOutput) override {
        // Simplified backward pass (not full BPTT)
        // Full implementation would unroll through time
        
        std::vector<double> gradInput(inputSize, 0.0);
        
        // Compute gradients (simplified)
        for (size_t i = 0; i < hiddenSize; ++i) {
            double delta = gradOutput[i];
            
            // Gradient w.r.t. input
            for (size_t j = 0; j < inputSize; ++j) {
                gradInput[j] += delta * Wxh[i][j];
                dWxh[i][j] += delta * cachedInput[j];
            }
            
            // Gradient w.r.t. hidden weights
            for (size_t j = 0; j < hiddenSize; ++j) {
                dWhh[i][j] += delta * hiddenState[j];
                dWhy[i][j] += delta * hiddenState[j];
            }
            
            dbh[i] += delta;
            dby[i] += delta;
        }
        
        return gradInput;
    }
    
    void update(double lr, double momentum = 0.0) {
        // Plain SGD update
        for (size_t i = 0; i < hiddenSize; ++i) {
            bh[i] -= lr * dbh[i];
            by[i] -= lr * dby[i];
            
            for (size_t j = 0; j < inputSize; ++j) {
                Wxh[i][j] -= lr * dWxh[i][j];
            }
            
            for (size_t j = 0; j < hiddenSize; ++j) {
                Whh[i][j] -= lr * dWhh[i][j];
                Why[i][j] -= lr * dWhy[i][j];
            }
        }
        
        // Zero gradients
        std::fill(dbh.begin(), dbh.end(), 0.0);
        std::fill(dby.begin(), dby.end(), 0.0);
        for (auto& row : dWxh) std::fill(row.begin(), row.end(), 0.0);
        for (auto& row : dWhh) std::fill(row.begin(), row.end(), 0.0);
        for (auto& row : dWhy) std::fill(row.begin(), row.end(), 0.0);
    }
    
    std::string type() const override { return "rnn"; }
    size_t outputSize() const override { return hiddenSize; }
    size_t paramCount() const override {
        return inputSize * hiddenSize + hiddenSize * hiddenSize + 
               hiddenSize * hiddenSize + 2 * hiddenSize;
    }
    
    std::string toJson() const override {
        return "{\"type\":\"rnn\",\"input_size\":" + std::to_string(inputSize) +
               ",\"hidden_size\":" + std::to_string(hiddenSize) + "}";
    }
};

std::mt19937 RNNLayer::rng(std::random_device{}());

} // namespace core
} // namespace neuraltrainer

#endif // NEURALTRAINER_RNN_LAYER_H
