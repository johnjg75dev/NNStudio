"""
app/modules/optimizers/optimizer_descriptors.py
UI-facing descriptor modules for each optimizer.
These carry tooltip/description data for the frontend.
Actual optimizer logic lives in app/core/optimizers.py.
"""
from ..base import BaseModule


class OptimizerDescriptor(BaseModule):
    category    = "optimizers"
    lr_range:   str = ""
    pros:       str = ""
    cons:       str = ""

    def to_dict(self) -> dict:
        return {
            "key":         self.key,
            "label":       self.label,
            "description": self.description,
            "category":    self.category,
            "lr_range":    self.lr_range,
            "pros":        self.pros,
            "cons":        self.cons,
        }


class SGDDescriptor(OptimizerDescriptor):
    key         = "sgd"
    label       = "SGD"
    description = "Vanilla stochastic gradient descent. Simple, predictable."
    lr_range    = "0.001 – 0.1"
    pros        = "Simple; no memory overhead; interpretable"
    cons        = "Sensitive to LR; slow on saddle points; no adaptivity"


class MomentumDescriptor(OptimizerDescriptor):
    key         = "momentum"
    label       = "SGD + Momentum"
    description = "SGD with exponential moving average. Smooths oscillations."
    lr_range    = "0.001 – 0.1"
    pros        = "Faster than SGD through valleys; dampens oscillations"
    cons        = "Extra momentum hyperparameter; can overshoot"


class RMSPropDescriptor(OptimizerDescriptor):
    key         = "rmsprop"
    label       = "RMSProp"
    description = "Divides LR by RMS of recent gradients — adaptive per-param."
    lr_range    = "0.0001 – 0.01"
    pros        = "Good for RNNs and non-stationary objectives"
    cons        = "No bias correction; less popular than Adam now"


class AdamDescriptor(OptimizerDescriptor):
    key         = "adam"
    label       = "Adam"
    description = "Adaptive Moment Estimation. Best all-round default."
    lr_range    = "0.0001 – 0.01"
    pros        = "Fast convergence; low sensitivity to LR; bias-corrected"
    cons        = "Can overfit on small datasets; slightly worse generalisation than AdamW"


class AdamWDescriptor(OptimizerDescriptor):
    key         = "adamw"
    label       = "AdamW"
    description = "Adam with decoupled weight decay. Best for generalisation."
    lr_range    = "0.0001 – 0.01"
    pros        = "Better generalisation than Adam; proper L2 regularisation"
    cons        = "Requires tuning weight_decay; marginal overhead"
