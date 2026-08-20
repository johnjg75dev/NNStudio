// Losses implementation file
// Registry initialization for loss functions

#include "core/losses.h"

namespace neuraltrainer {
namespace core {

// Loss registry static members
std::unordered_map<std::string, std::unique_ptr<LossFunction>> LossRegistry::registry;
bool LossRegistry::initialized = false;

} // namespace core
} // namespace neuraltrainer
