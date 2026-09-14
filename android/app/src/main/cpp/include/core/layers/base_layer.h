#ifndef NEURALTRAINER_BASE_LAYER_H
#define NEURALTRAINER_BASE_LAYER_H

#include <vector>
#include <string>
#include <memory>
#include <random>

namespace neuraltrainer {
namespace core {

/**
 * Base class for all neural network layers
 */
class Layer {
protected:
    // Cached input from forward pass (needed for backprop)
    std::vector<double> cachedInput;
    std::vector<double> cachedOutput;
    
    // Pre-activation values (before activation function)
    std::vector<double> preActivation;
    
public:
    virtual ~Layer() = default;
    
    /**
     * Forward pass
     * @param input Input vector
     * @param training Whether in training mode (affects dropout, batchnorm, etc.)
     * @return Output vector
     */
    virtual std::vector<double> forward(const std::vector<double>& input, bool training = true) = 0;
    
    /**
     * Backward pass - compute gradients
     * @param gradOutput Gradient flowing from the next layer
     * @return Gradient w.r.t. input (to flow to previous layer)
     */
    virtual std::vector<double> backward(const std::vector<double>& gradOutput) = 0;
    
    /**
     * Get layer type name
     */
    virtual std::string type() const = 0;
    
    /**
     * Get number of output features
     */
    virtual size_t outputSize() const = 0;
    
    /**
     * Get number of parameters in this layer
     */
    virtual size_t paramCount() const { return 0; }
    
    /**
     * Serialize layer to JSON-like structure (for saving/loading models)
     */
    virtual std::string toJson() const = 0;
};

} // namespace core
} // namespace neuraltrainer

#endif // NEURALTRAINER_BASE_LAYER_H
