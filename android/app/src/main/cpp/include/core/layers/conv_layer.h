// Convolutional Layer header
#ifndef NEURALTRAINER_CONV_LAYER_H
#define NEURALTRAINER_CONV_LAYER_H

#include "base_layer.h"
#include "../activations.h"
#include <vector>
#include <string>
#include <random>
#include <cmath>

namespace neuraltrainer {
namespace core {

/**
 * 2D Convolutional Layer
 * Supports configurable kernel size, stride, and padding
 */
class ConvLayer : public Layer {
private:
    size_t inChannels;
    size_t outChannels;
    size_t kernelSize;
    size_t stride;
    size_t padding;
    size_t inputHeight;
    size_t inputWidth;
    
    // Weights: [outChannels][inChannels][kernelSize][kernelSize]
    std::vector<std::vector<std::vector<std::vector<double>>>> W;
    
    // Biases: [outChannels]
    std::vector<double> b;
    
    // Gradients
    std::vector<std::vector<std::vector<std::vector<double>>>> dW;
    std::vector<double> db;
    
    const Activation* activation;
    
    static std::mt19937 rng;
    
public:
    /**
     * Constructor for 2D Conv layer
     * @param inC Number of input channels
     * @param outC Number of output channels (filters)
     * @param kSize Kernel size (assumes square kernel)
     * @param stride Stride for convolution
     * @param pad Padding amount
     * @param act Activation function
     */
    ConvLayer(size_t inC, size_t outC, size_t kSize, size_t stride = 1, 
              size_t pad = 0, const Activation* act = nullptr)
        : inChannels(inC), outChannels(outC), kernelSize(kSize),
          stride(stride), padding(pad), activation(act) {
        
        if (!activation) {
            activation = ActivationRegistry::get("relu");
        }
        
        // Initialize weights with He initialization
        double scale = std::sqrt(2.0 / (inChannels * kernelSize * kernelSize));
        std::uniform_real_distribution<double> dist(-scale, scale);
        
        W.resize(outChannels, 
            std::vector<std::vector<std::vector<double>>>(inChannels,
                std::vector<std::vector<double>>(kernelSize,
                    std::vector<double>(kernelSize, 0.0))));
        
        b.resize(outChannels, 0.0);
        
        dW.resize(outChannels,
            std::vector<std::vector<std::vector<double>>>(inChannels,
                std::vector<std::vector<double>>(kernelSize,
                    std::vector<double>(kernelSize, 0.0))));
        
        db.resize(outChannels, 0.0);
        
        // Initialize weights
        for (size_t oc = 0; oc < outChannels; ++oc) {
            for (size_t ic = 0; ic < inChannels; ++ic) {
                for (size_t ki = 0; ki < kernelSize; ++ki) {
                    for (size_t kj = 0; kj < kernelSize; ++kj) {
                        W[oc][ic][ki][kj] = dist(rng);
                    }
                }
            }
        }
    }
    
    std::vector<double> forward(const std::vector<double>& input, bool training = true) override {
        // For simplicity, assume input is flattened [batch, channels, height, width]
        // In a full implementation, we'd handle proper tensor shapes
        
        // Placeholder - returns input unchanged
        // Full implementation would perform actual convolution
        cachedInput = input;
        cachedOutput = input;  // TODO: Implement proper conv forward
        
        return cachedOutput;
    }
    
    std::vector<double> backward(const std::vector<double>& gradOutput) override {
        // Placeholder - returns gradient unchanged
        // Full implementation would compute proper conv gradients
        return gradOutput;
    }
    
    void update(double lr, double momentum = 0.0,
                std::vector<std::vector<std::vector<std::vector<double>>>>* velW = nullptr,
                std::vector<double>* velB = nullptr) {
        
        if (velW && velB) {
            // SGD with momentum
            for (size_t oc = 0; oc < outChannels; ++oc) {
                db[oc] = momentum * (*velB)[oc] - lr * db[oc];
                b[oc] += db[oc];
                
                for (size_t ic = 0; ic < inChannels; ++ic) {
                    for (size_t ki = 0; ki < kernelSize; ++ki) {
                        for (size_t kj = 0; kj < kernelSize; ++kj) {
                            (*velW)[oc][ic][ki][kj] = momentum * (*velW)[oc][ic][ki][kj] - lr * dW[oc][ic][ki][kj];
                            W[oc][ic][ki][kj] += (*velW)[oc][ic][ki][kj];
                        }
                    }
                }
            }
        } else {
            // Plain SGD
            for (size_t oc = 0; oc < outChannels; ++oc) {
                b[oc] -= lr * db[oc];
                for (size_t ic = 0; ic < inChannels; ++ic) {
                    for (size_t ki = 0; ki < kernelSize; ++ki) {
                        for (size_t kj = 0; kj < kernelSize; ++kj) {
                            W[oc][ic][ki][kj] -= lr * dW[oc][ic][ki][kj];
                        }
                    }
                }
            }
        }
    }
    
    std::string type() const override { return "conv2d"; }
    size_t outputSize() const override { return outChannels; }
    size_t paramCount() const override { 
        return inChannels * outChannels * kernelSize * kernelSize + outChannels; 
    }
    
    std::string toJson() const override {
        return "{\"type\":\"conv2d\",\"in_channels\":" + std::to_string(inChannels) +
               ",\"out_channels\":" + std::to_string(outChannels) +
               ",\"kernel_size\":" + std::to_string(kernelSize) +
               ",\"stride\":" + std::to_string(stride) +
               ",\"padding\":" + std::to_string(padding) + "}";
    }
};

std::mt19937 ConvLayer::rng(std::random_device{}());

} // namespace core
} // namespace neuraltrainer

#endif // NEURALTRAINER_CONV_LAYER_H
