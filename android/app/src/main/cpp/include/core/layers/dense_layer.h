#ifndef NEURALTRAINER_DENSE_LAYER_H
#define NEURALTRAINER_DENSE_LAYER_H

#include "base_layer.h"
#include "../activations.h"
#include <vector>
#include <string>
#include <random>
#include <sstream>
#include <iomanip>

namespace neuraltrainer {
namespace core {

/**
 * Fully Connected (Dense) Layer
 * output = activation(W * input + b)
 */
class DenseLayer : public Layer {
private:
    size_t nIn;
    size_t nOut;
    
    // Weights: W[i][j] connects input j to output i
    std::vector<std::vector<double>> W;
    
    // Biases: b[i] for output neuron i
    std::vector<double> b;
    
    // Gradients
    std::vector<std::vector<double>> dW;
    std::vector<double> db;
    
    // Activation function
    const Activation* activation;
    
    // Whether this is the output layer (affects gradient computation)
    bool isOutput;
    
    // Random number generator for initialization
    static std::mt19937 rng;
    
public:
    /**
     * Constructor
     * @param inputSize Number of input features
     * @param outputSize Number of output neurons
     * @param act Activation function (default: Tanh)
     * @param output Whether this is the output layer
     */
    DenseLayer(size_t inputSize, size_t outputSize, 
               const Activation* act = nullptr, bool output = false)
        : nIn(inputSize), nOut(outputSize), activation(act), isOutput(output) {
        
        if (!activation) {
            activation = ActivationRegistry::get("tanh");
        }
        
        // Initialize weights using He initialization
        double scale = std::sqrt(2.0 / nIn);
        
        std::uniform_real_distribution<double> dist(-scale, scale);
        
        W.resize(nOut, std::vector<double>(nIn));
        b.resize(nOut, 0.0);
        dW.resize(nOut, std::vector<double>(nIn, 0.0));
        db.resize(nOut, 0.0);
        
        // He initialization with uniform distribution
        for (size_t i = 0; i < nOut; ++i) {
            for (size_t j = 0; j < nIn; ++j) {
                W[i][j] = dist(rng);
            }
        }
    }
    
    std::vector<double> forward(const std::vector<double>& input, bool training = true) override {
        cachedInput = input;
        preActivation.resize(nOut);
        cachedOutput.resize(nOut);
        
        // Compute z = W * x + b
        for (size_t i = 0; i < nOut; ++i) {
            double sum = b[i];
            for (size_t j = 0; j < nIn; ++j) {
                sum += W[i][j] * input[j];
            }
            preActivation[i] = sum;
            
            // Apply activation
            if (isOutput) {
                // Output layer always uses sigmoid for classification
                cachedOutput[i] = 1.0 / (1.0 + std::exp(-sum));
            } else {
                cachedOutput[i] = activation->forward(sum);
            }
        }
        
        return cachedOutput;
    }
    
    std::vector<double> backward(const std::vector<double>& gradOutput) override {
        // gradOutput is dL/d(output) from the next layer
        
        // For output layer, gradOutput already includes loss gradient
        // For hidden layers, we need to apply activation derivative
        
        std::vector<double> delta(nOut);
        
        if (isOutput) {
            // Output layer with sigmoid: combined gradient is (pred - target)
            // gradOutput already contains this
            delta = gradOutput;
        } else {
            // Hidden layer: apply activation derivative
            for (size_t i = 0; i < nOut; ++i) {
                delta[i] = gradOutput[i] * activation->backward(preActivation[i]);
            }
        }
        
        // Compute weight gradients: dW[i][j] = delta[i] * input[j]
        for (size_t i = 0; i < nOut; ++i) {
            db[i] = delta[i];  // Bias gradient
            for (size_t j = 0; j < nIn; ++j) {
                dW[i][j] = delta[i] * cachedInput[j];
            }
        }
        
        // Compute gradient w.r.t. input: dL/dx[j] = sum_i(delta[i] * W[i][j])
        std::vector<double> gradInput(nIn, 0.0);
        for (size_t j = 0; j < nIn; ++j) {
            for (size_t i = 0; i < nOut; ++i) {
                gradInput[j] += delta[i] * W[i][j];
            }
        }
        
        return gradInput;
    }
    
    /**
     * Update weights using optimizer
     * @param lr Learning rate
     * @param momentum Momentum coefficient (for SGD with momentum)
     * @param velocity Velocity vectors for momentum-based optimizers
     */
    void update(double lr, double momentum = 0.0,
                std::vector<std::vector<double>>* velW = nullptr,
                std::vector<double>* velB = nullptr) {
        
        if (velW && velB) {
            // SGD with momentum
            for (size_t i = 0; i < nOut; ++i) {
                (*velB)[i] = momentum * (*velB)[i] - lr * db[i];
                b[i] += (*velB)[i];
                
                for (size_t j = 0; j < nIn; ++j) {
                    (*velW)[i][j] = momentum * (*velW)[i][j] - lr * dW[i][j];
                    W[i][j] += (*velW)[i][j];
                }
            }
        } else {
            // Plain SGD
            for (size_t i = 0; i < nOut; ++i) {
                b[i] -= lr * db[i];
                for (size_t j = 0; j < nIn; ++j) {
                    W[i][j] -= lr * dW[i][j];
                }
            }
        }
    }
    
    std::string type() const override { return "dense"; }
    size_t outputSize() const override { return nOut; }
    size_t paramCount() const override { return nIn * nOut + nOut; }
    
    std::string toJson() const override {
        std::ostringstream oss;
        oss << "{\"type\":\"dense\",\"n_in\":" << nIn 
            << ",\"n_out\":" << nOut 
            << ",\"activation\":\"" << (isOutput ? "sigmoid" : activation->name()) << "\"}";
        return oss.str();
    }
    
    // Getters for visualization and serialization
    const std::vector<std::vector<double>>& getWeights() const { return W; }
    const std::vector<double>& getBiases() const { return b; }
    const std::vector<double>& getOutput() const { return cachedOutput; }
};

// Static member initialization
std::mt19937 DenseLayer::rng(std::random_device{}());

} // namespace core
} // namespace neuraltrainer

#endif // NEURALTRAINER_DENSE_LAYER_H
