#ifndef NEURALTRAINER_LOSSES_H
#define NEURALTRAINER_LOSSES_H

#include <cmath>
#include <string>
#include <unordered_map>
#include <memory>
#include <vector>

namespace neuraltrainer {
namespace core {

/**
 * Loss function interface
 */
class LossFunction {
public:
    virtual ~LossFunction() = default;
    
    /**
     * Compute loss between prediction and target
     */
    virtual double compute(double pred, double target) const = 0;
    
    /**
     * Compute gradient of loss w.r.t. prediction
     * This is d(loss)/d(pred), used in backpropagation
     */
    virtual double gradient(double pred, double target) const = 0;
    
    /**
     * Get loss function name
     */
    virtual std::string name() const = 0;
};

/**
 * Mean Squared Error (MSE) Loss: (pred - target)^2
 * Gradient: 2 * (pred - target)
 */
class MSELoss : public LossFunction {
public:
    double compute(double pred, double target) const override {
        double diff = pred - target;
        return diff * diff;
    }
    
    double gradient(double pred, double target) const override {
        return 2.0 * (pred - target);
    }
    
    std::string name() const override { return "mse"; }
};

/**
 * Binary Cross-Entropy (BCE) Loss: -[target*log(pred) + (1-target)*log(1-pred)]
 * Assumes pred is already sigmoid-activated (in [0, 1])
 * Gradient: (pred - target) / (pred * (1 - pred))
 * Simplified when combined with sigmoid: pred - target
 */
class BCELoss : public LossFunction {
private:
    static constexpr double EPS = 1e-15;
    
public:
    double compute(double pred, double target) const override {
        double p = std::max(EPS, std::min(1.0 - EPS, pred));
        return -(target * std::log(p) + (1.0 - target) * std::log(1.0 - p));
    }
    
    double gradient(double pred, double target) const override {
        // When BCE is combined with sigmoid activation, the gradient simplifies
        // to just (pred - target). This is the form we use for efficiency.
        return pred - target;
    }
    
    std::string name() const override { return "bce"; }
};

/**
 * Mean Absolute Error (MAE) Loss: |pred - target|
 * Gradient: sign(pred - target)
 */
class MAELoss : public LossFunction {
public:
    double compute(double pred, double target) const override {
        return std::abs(pred - target);
    }
    
    double gradient(double pred, double target) const override {
        double diff = pred - target;
        if (diff > 0) return 1.0;
        if (diff < 0) return -1.0;
        return 0.0;
    }
    
    std::string name() const override { return "mae"; }
};

/**
 * Categorical Cross-Entropy Loss (for multi-class classification)
 * Assumes pred is softmax output and target is one-hot encoded
 */
class CategoricalCrossEntropy : public LossFunction {
private:
    static constexpr double EPS = 1e-15;
    
public:
    double compute(double pred, double target) const override {
        double p = std::max(EPS, pred);
        return -target * std::log(p);
    }
    
    double gradient(double pred, double target) const override {
        // Combined with softmax: gradient is (pred - target)
        return pred - target;
    }
    
    std::string name() const override { return "categorical_crossentropy"; }
};

/**
 * Huber Loss: Robust loss that's MSE for small errors, MAE for large errors
 */
class HuberLoss : public LossFunction {
private:
    double delta;
    
public:
    explicit HuberLoss(double d = 1.0) : delta(d) {}
    
    double compute(double pred, double target) const override {
        double diff = pred - target;
        double abs_diff = std::abs(diff);
        if (abs_diff <= delta) {
            return 0.5 * diff * diff;
        } else {
            return delta * (abs_diff - 0.5 * delta);
        }
    }
    
    double gradient(double pred, double target) const override {
        double diff = pred - target;
        double abs_diff = std::abs(diff);
        if (abs_diff <= delta) {
            return diff;
        } else {
            return delta * (diff > 0 ? 1.0 : -1.0);
        }
    }
    
    std::string name() const override { return "huber"; }
};

/**
 * Loss Registry - provides access to all loss functions by name
 */
class LossRegistry {
private:
    static std::unordered_map<std::string, std::unique_ptr<LossFunction>> registry;
    static bool initialized;
    
    static void init() {
        if (!initialized) {
            registry["mse"] = std::make_unique<MSELoss>();
            registry["bce"] = std::make_unique<BCELoss>();
            registry["mae"] = std::make_unique<MAELoss>();
            registry["categorical_crossentropy"] = std::make_unique<CategoricalCrossEntropy>();
            registry["huber"] = std::make_unique<HuberLoss>();
            initialized = true;
        }
    }
    
public:
    static const LossFunction* get(const std::string& name) {
        init();
        auto it = registry.find(name);
        return it != registry.end() ? it->second.get() : nullptr;
    }
};

} // namespace core
} // namespace neuraltrainer

#endif // NEURALTRAINER_LOSSES_H
