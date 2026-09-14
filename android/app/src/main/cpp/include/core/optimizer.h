#ifndef NEURALTRAINER_OPTIMIZER_H
#define NEURALTRAINER_OPTIMIZER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <cmath>

namespace neuraltrainer {
namespace core {

/**
 * Base optimizer interface
 */
class Optimizer {
protected:
    double lr;
    double weightDecay;
    
public:
    explicit Optimizer(double learningRate = 0.01, double wd = 0.0)
        : lr(learningRate), weightDecay(wd) {}
    
    virtual ~Optimizer() = default;
    
    /**
     * Update a single parameter
     * @param param Reference to the parameter to update
     * @param grad Gradient of the parameter
     * @param key Unique identifier for stateful optimizers (Adam, RMSProp)
     */
    virtual void update(double& param, double grad, const std::string& key = "") = 0;
    
    /**
     * Get optimizer name
     */
    virtual std::string name() const = 0;
    
    void setLearningRate(double newLr) { lr = newLr; }
    double getLearningRate() const { return lr; }
};

/**
 * SGD (Stochastic Gradient Descent)
 * param = param - lr * grad
 */
class SGDOptimizer : public Optimizer {
public:
    using Optimizer::Optimizer;
    
    void update(double& param, double grad, const std::string& /*key*/) override {
        // Apply weight decay if specified
        if (weightDecay > 0) {
            grad += weightDecay * param;
        }
        param -= lr * grad;
    }
    
    std::string name() const override { return "sgd"; }
};

/**
 * SGD with Momentum
 * v = momentum * v - lr * grad
 * param = param + v
 */
class MomentumOptimizer : public Optimizer {
private:
    double momentum;
    std::unordered_map<std::string, double> velocity;
    
public:
    MomentumOptimizer(double learningRate = 0.01, double mom = 0.9, double wd = 0.0)
        : Optimizer(learningRate, wd), momentum(mom) {}
    
    void update(double& param, double grad, const std::string& key) override {
        if (weightDecay > 0) {
            grad += weightDecay * param;
        }
        
        double& v = velocity[key];
        v = momentum * v - lr * grad;
        param += v;
    }
    
    std::string name() const override { return "momentum"; }
};

/**
 * RMSProp Optimizer
 * v = rho * v + (1 - rho) * grad^2
 * param = param - lr * grad / sqrt(v + eps)
 */
class RMSPropOptimizer : public Optimizer {
private:
    double rho;
    static constexpr double eps = 1e-8;
    std::unordered_map<std::string, double> cache;
    
public:
    RMSPropOptimizer(double learningRate = 0.001, double r = 0.9, double wd = 0.0)
        : Optimizer(learningRate, wd), rho(r) {}
    
    void update(double& param, double grad, const std::string& key) override {
        if (weightDecay > 0) {
            grad += weightDecay * param;
        }
        
        double& v = cache[key];
        v = rho * v + (1.0 - rho) * grad * grad;
        param -= lr * grad / (std::sqrt(v) + eps);
    }
    
    std::string name() const override { return "rmsprop"; }
};

/**
 * Adam Optimizer
 * m = beta1 * m + (1 - beta1) * grad
 * v = beta2 * v + (1 - beta2) * grad^2
 * m_hat = m / (1 - beta1^t)
 * v_hat = v / (1 - beta2^t)
 * param = param - lr * m_hat / (sqrt(v_hat) + eps)
 */
class AdamOptimizer : public Optimizer {
private:
    double beta1, beta2;
    static constexpr double eps = 1e-8;
    int timestep;
    
    struct State {
        double m = 0.0;  // First moment
        double v = 0.0;  // Second moment
    };
    
    std::unordered_map<std::string, State> state;
    
public:
    AdamOptimizer(double learningRate = 0.001, double b1 = 0.9, double b2 = 0.999, double wd = 0.0)
        : Optimizer(learningRate, wd), beta1(b1), beta2(b2), timestep(0) {}
    
    void update(double& param, double grad, const std::string& key) override {
        if (timestep == 0) {
            timestep = 1;  // Start from 1 for bias correction
        }
        
        if (weightDecay > 0) {
            grad += weightDecay * param;
        }
        
        State& s = state[key];
        
        // Update biased first moment estimate
        s.m = beta1 * s.m + (1.0 - beta1) * grad;
        
        // Update biased second raw moment estimate
        s.v = beta2 * s.v + (1.0 - beta2) * grad * grad;
        
        // Compute bias-corrected first moment estimate
        double mHat = s.m / (1.0 - std::pow(beta1, timestep));
        
        // Compute bias-corrected second raw moment estimate
        double vHat = s.v / (1.0 - std::pow(beta2, timestep));
        
        // Update parameter
        param -= lr * mHat / (std::sqrt(vHat) + eps);
        
        timestep++;
    }
    
    void reset() {
        timestep = 0;
        state.clear();
    }
    
    std::string name() const override { return "adam"; }
};

/**
 * AdamW Optimizer (Decoupled Weight Decay)
 * Same as Adam, but weight decay is applied separately (more principled)
 */
class AdamWOptimizer : public Optimizer {
private:
    double beta1, beta2;
    static constexpr double eps = 1e-8;
    int timestep;
    
    struct State {
        double m = 0.0;
        double v = 0.0;
    };
    
    std::unordered_map<std::string, State> state;
    
public:
    AdamWOptimizer(double learningRate = 0.001, double b1 = 0.9, double b2 = 0.999, double wd = 0.0)
        : Optimizer(learningRate, wd), beta1(b1), beta2(b2), timestep(0) {}
    
    void update(double& param, double grad, const std::string& key) override {
        if (timestep == 0) {
            timestep = 1;
        }
        
        State& s = state[key];
        
        // Update biased first moment estimate
        s.m = beta1 * s.m + (1.0 - beta1) * grad;
        
        // Update biased second raw moment estimate
        s.v = beta2 * s.v + (1.0 - beta2) * grad * grad;
        
        // Compute bias-corrected estimates
        double mHat = s.m / (1.0 - std::pow(beta1, timestep));
        double vHat = s.v / (1.0 - std::pow(beta2, timestep));
        
        // Adam update
        param -= lr * mHat / (std::sqrt(vHat) + eps);
        
        // Decoupled weight decay
        if (weightDecay > 0) {
            param -= lr * weightDecay * param;
        }
        
        timestep++;
    }
    
    void reset() {
        timestep = 0;
        state.clear();
    }
    
    std::string name() const override { return "adamw"; }
};

/**
 * Optimizer Factory
 */
class OptimizerFactory {
public:
    static std::unique_ptr<Optimizer> create(const std::string& name, 
                                             double lr = 0.01,
                                             double momentum = 0.9,
                                             double weightDecay = 0.0) {
        if (name == "sgd") {
            return std::make_unique<SGDOptimizer>(lr, weightDecay);
        } else if (name == "momentum") {
            return std::make_unique<MomentumOptimizer>(lr, momentum, weightDecay);
        } else if (name == "rmsprop") {
            return std::make_unique<RMSPropOptimizer>(lr, 0.9, weightDecay);
        } else if (name == "adam") {
            return std::make_unique<AdamOptimizer>(lr, 0.9, 0.999, weightDecay);
        } else if (name == "adamw") {
            return std::make_unique<AdamWOptimizer>(lr, 0.9, 0.999, weightDecay);
        }
        
        // Default to Adam
        return std::make_unique<AdamOptimizer>(lr, 0.9, 0.999, weightDecay);
    }
};

} // namespace core
} // namespace neuraltrainer

#endif // NEURALTRAINER_OPTIMIZER_H
