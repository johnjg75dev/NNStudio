#ifndef NEURALTRAINER_ACTIVATIONS_H
#define NEURALTRAINER_ACTIVATIONS_H

#include <cmath>
#include <string>
#include <unordered_map>
#include <functional>

namespace neuraltrainer {
namespace core {

/**
 * Activation function interface
 */
class Activation {
public:
    virtual ~Activation() = default;
    
    /**
     * Forward pass: f(x)
     */
    virtual double forward(double x) const = 0;
    
    /**
     * Backward pass: f'(x) - derivative at point x
     * Note: x here is the pre-activation value
     */
    virtual double backward(double x) const = 0;
    
    /**
     * Get activation name
     */
    virtual std::string name() const = 0;
};

/**
 * ReLU Activation: max(0, x)
 */
class ReLU : public Activation {
public:
    double forward(double x) const override {
        return x > 0 ? x : 0;
    }
    
    double backward(double x) const override {
        return x > 0 ? 1.0 : 0.0;
    }
    
    std::string name() const override { return "relu"; }
};

/**
 * Leaky ReLU: max(alpha*x, x)
 */
class LeakyReLU : public Activation {
private:
    double alpha;
public:
    explicit LeakyReLU(double a = 0.01) : alpha(a) {}
    
    double forward(double x) const override {
        return x > 0 ? x : alpha * x;
    }
    
    double backward(double x) const override {
        return x > 0 ? 1.0 : alpha;
    }
    
    std::string name() const override { return "leaky_relu"; }
};

/**
 * Tanh Activation: (e^x - e^-x) / (e^x + e^-x)
 */
class Tanh : public Activation {
public:
    double forward(double x) const override {
        return std::tanh(x);
    }
    
    double backward(double x) const override {
        double t = std::tanh(x);
        return 1.0 - t * t;
    }
    
    std::string name() const override { return "tanh"; }
};

/**
 * Sigmoid Activation: 1 / (1 + e^-x)
 */
class Sigmoid : public Activation {
public:
    double forward(double x) const override {
        return 1.0 / (1.0 + std::exp(-x));
    }
    
    double backward(double x) const override {
        double s = 1.0 / (1.0 + std::exp(-x));
        return s * (1.0 - s);
    }
    
    std::string name() const override { return "sigmoid"; }
};

/**
 * GELU Activation: x * Φ(x) where Φ is Gaussian CDF
 * Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
 */
class GELU : public Activation {
public:
    double forward(double x) const override {
        constexpr double sqrt_2_pi = 0.7978845608028654;
        return 0.5 * x * (1.0 + std::tanh(sqrt_2_pi * (x + 0.044715 * x * x * x)));
    }
    
    double backward(double x) const override {
        constexpr double sqrt_2_pi = 0.7978845608028654;
        double x2 = x * x;
        double x3 = x2 * x;
        double tanh_arg = sqrt_2_pi * (x + 0.044715 * x3);
        double tanh_val = std::tanh(tanh_arg);
        double sech_sq = 1.0 - tanh_val * tanh_val;
        return 0.5 * (1.0 + tanh_val) + 0.5 * x * sqrt_2_pi * (1.0 + 0.134145 * x2) * sech_sq;
    }
    
    std::string name() const override { return "gelu"; }
};

/**
 * Swish Activation: x * sigmoid(x)
 */
class Swish : public Activation {
public:
    double forward(double x) const override {
        double sig = 1.0 / (1.0 + std::exp(-x));
        return x * sig;
    }
    
    double backward(double x) const override {
        double sig = 1.0 / (1.0 + std::exp(-x));
        return sig + x * sig * (1.0 - sig);
    }
    
    std::string name() const override { return "swish"; }
};

/**
 * Linear/Identity Activation: f(x) = x
 */
class Linear : public Activation {
public:
    double forward(double x) const override { return x; }
    double backward(double x) const override { return 1.0; }
    std::string name() const override { return "linear"; }
};

/**
 * Activation Registry - provides access to all activations by name
 */
class ActivationRegistry {
private:
    static std::unordered_map<std::string, std::unique_ptr<Activation>> registry;
    static bool initialized;
    
    static void init() {
        if (!initialized) {
            registry["relu"] = std::make_unique<ReLU>();
            registry["leaky_relu"] = std::make_unique<LeakyReLU>();
            registry["tanh"] = std::make_unique<Tanh>();
            registry["sigmoid"] = std::make_unique<Sigmoid>();
            registry["gelu"] = std::make_unique<GELU>();
            registry["swish"] = std::make_unique<Swish>();
            registry["linear"] = std::make_unique<Linear>();
            initialized = true;
        }
    }
    
public:
    static const Activation* get(const std::string& name) {
        init();
        auto it = registry.find(name);
        return it != registry.end() ? it->second.get() : nullptr;
    }
};

} // namespace core
} // namespace neuraltrainer

#endif // NEURALTRAINER_ACTIVATIONS_H
