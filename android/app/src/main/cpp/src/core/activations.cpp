// Activations implementation file
// Most activations are header-only for performance, but this file
// ensures the symbols are properly exported in the shared library

#include "core/activations.h"

namespace neuraltrainer {
namespace core {

// Explicit template instantiations and registry initialization
std::unordered_map<std::string, std::unique_ptr<Activation>> ActivationRegistry::registry;
bool ActivationRegistry::initialized = false;

} // namespace core
} // namespace neuraltrainer
