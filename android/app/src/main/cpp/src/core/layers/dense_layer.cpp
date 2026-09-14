// Dense Layer implementation
// Most functionality is in the header for performance, but this file
// provides explicit template instantiations if needed

#include "core/layers/dense_layer.h"

namespace neuraltrainer {
namespace core {

// Dense layer methods are all inline in the header for performance
// This file exists for potential future expansions like:
// - NEON-optimized matrix multiplication
// - Multi-threaded forward/backward passes
// - Quantized weight storage

} // namespace core
} // namespace neuraltrainer
