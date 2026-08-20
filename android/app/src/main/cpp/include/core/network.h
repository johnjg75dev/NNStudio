#ifndef NEURALTRAINER_NETWORK_H
#define NEURALTRAINER_NETWORK_H

#include "layers/dense_layer.h"
#include "optimizer.h"
#include "losses.h"
#include <vector>
#include <memory>
#include <string>
#include <sstream>
#include <iomanip>

namespace neuraltrainer {
namespace core {

/**
 * Neural Network container and trainer
 */
class NeuralNetwork {
private:
    std::vector<std::unique_ptr<Layer>> layers;
    std::unique_ptr<Optimizer> optimizer;
    const LossFunction* lossFn;
    
    int epoch;
    std::vector<double> lossHistory;
    
public:
    /**
     * Constructor
     * @param layerList List of layer pointers (network takes ownership)
     * @param opt Optimizer to use for training
     * @param loss Loss function
     */
    NeuralNetwork(std::vector<std::unique_ptr<Layer>> layerList,
                  std::unique_ptr<Optimizer> opt,
                  const LossFunction* loss)
        : layers(std::move(layerList))
        , optimizer(std::move(opt))
        , lossFn(loss)
        , epoch(0) {}
    
    /**
     * Forward pass through the network
     * @param input Input vector
     * @param training Whether in training mode
     * @return Output vector
     */
    std::vector<double> predict(const std::vector<double>& input, bool training = false) {
        std::vector<double> current = input;
        for (auto& layer : layers) {
            current = layer->forward(current, training);
        }
        return current;
    }
    
    /**
     * Single training step on one sample
     * @param x Input vector
     * @param y Target vector
     * @return Loss value for this sample
     */
    double trainStep(const std::vector<double>& x, const std::vector<double>& y) {
        // Forward pass
        std::vector<double> output = predict(x, true);
        
        // Compute loss
        double totalLoss = 0.0;
        std::vector<double> gradOutput(output.size());
        for (size_t i = 0; i < output.size(); ++i) {
            totalLoss += lossFn->compute(output[i], y[i]);
            gradOutput[i] = lossFn->gradient(output[i], y[i]);
        }
        
        // Backward pass through layers in reverse
        std::vector<double> delta = gradOutput;
        for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i) {
            auto* denseLayer = dynamic_cast<DenseLayer*>(layers[i].get());
            if (denseLayer) {
                delta = denseLayer->backward(delta);
                
                // Update weights
                if (optimizer->name() == "momentum") {
                    // Momentum optimizer needs velocity tracking
                    // For simplicity, we'll use plain SGD update here
                    // A full implementation would track velocity per parameter
                    denseLayer->update(optimizer->getLearningRate());
                } else {
                    denseLayer->update(optimizer->getLearningRate());
                }
            }
        }
        
        return totalLoss / output.size();
    }
    
    /**
     * Train for one epoch over the dataset
     * @param dataset Training data as list of {input, target} pairs
     * @param lr Optional learning rate override
     * @return Average loss for the epoch
     */
    double trainEpoch(const std::vector<std::pair<std::vector<double>, std::vector<double>>>& dataset,
                      double lr = -1.0) {
        if (lr > 0) {
            optimizer->setLearningRate(lr);
        }
        
        double totalLoss = 0.0;
        
        // Shuffle indices
        std::vector<size_t> indices(dataset.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            indices[i] = i;
        }
        
        // Simple shuffle using random device
        static std::mt19937 g(std::random_device{}());
        for (size_t i = indices.size() - 1; i > 0; --i) {
            std::uniform_int_distribution<size_t> dist(0, i);
            size_t j = dist(g);
            std::swap(indices[i], indices[j]);
        }
        
        // Train on each sample
        for (size_t idx : indices) {
            totalLoss += trainStep(dataset[idx].first, dataset[idx].second);
        }
        
        epoch++;
        double avgLoss = totalLoss / dataset.size();
        lossHistory.push_back(avgLoss);
        
        return avgLoss;
    }
    
    /**
     * Compute average loss over dataset
     */
    double computeLoss(const std::vector<std::pair<std::vector<double>, std::vector<double>>>& dataset) {
        double totalLoss = 0.0;
        for (const auto& sample : dataset) {
            std::vector<double> output = predict(sample.first, false);
            for (size_t i = 0; i < output.size(); ++i) {
                totalLoss += lossFn->compute(output[i], sample.second[i]);
            }
        }
        return totalLoss / dataset.size();
    }
    
    /**
     * Compute accuracy over dataset (for classification)
     */
    double computeAccuracy(const std::vector<std::pair<std::vector<double>, std::vector<double>>>& dataset,
                           double threshold = 0.5) {
        int correct = 0;
        for (const auto& sample : dataset) {
            std::vector<double> output = predict(sample.first, false);
            bool allCorrect = true;
            for (size_t i = 0; i < output.size(); ++i) {
                double pred = output[i] > threshold ? 1.0 : 0.0;
                if (std::abs(pred - sample.second[i]) > 0.5) {
                    allCorrect = false;
                    break;
                }
            }
            if (allCorrect) correct++;
        }
        return static_cast<double>(correct) / dataset.size();
    }
    
    /**
     * Get network topology (layer sizes)
     */
    std::vector<size_t> getTopology() const {
        std::vector<size_t> topo;
        if (layers.empty()) return topo;
        
        // Get input size from first layer
        auto* firstDense = dynamic_cast<const DenseLayer*>(layers[0].get());
        if (firstDense) {
            // We'd need a getter for input size - for now just track outputs
        }
        
        // Add output size for each layer
        for (const auto& layer : layers) {
            topo.push_back(layer->outputSize());
        }
        
        return topo;
    }
    
    /**
     * Get total parameter count
     */
    size_t getParamCount() const {
        size_t total = 0;
        for (const auto& layer : layers) {
            total += layer->paramCount();
        }
        return total;
    }
    
    /**
     * Get activations at each layer for visualization
     */
    std::vector<std::vector<double>> getActivations(const std::vector<double>& input) {
        std::vector<std::vector<double>> snaps;
        snaps.push_back(input);  // Input layer
        
        std::vector<double> current = input;
        for (auto& layer : layers) {
            current = layer->forward(current, false);
            
            auto* denseLayer = dynamic_cast<DenseLayer*>(layer.get());
            if (denseLayer) {
                snaps.push_back(denseLayer->getOutput());
            } else {
                snaps.push_back(current);
            }
        }
        
        return snaps;
    }
    
    /**
     * Serialize network to JSON
     */
    std::string toJson() const {
        std::ostringstream oss;
        oss << "{\n";
        oss << "  \"epoch\": " << epoch << ",\n";
        oss << "  \"optimizer\": \"" << optimizer->name() << "\",\n";
        oss << "  \"learning_rate\": " << std::fixed << std::setprecision(6) << optimizer->getLearningRate() << ",\n";
        oss << "  \"loss\": \"" << lossFn->name() << "\",\n";
        oss << "  \"layers\": [";
        
        for (size_t i = 0; i < layers.size(); ++i) {
            if (i > 0) oss << ",";
            oss << "\n    " << layers[i]->toJson();
        }
        
        oss << "\n  ],\n";
        oss << "  \"loss_history\": [";
        
        // Keep last 200 entries
        size_t start = lossHistory.size() > 200 ? lossHistory.size() - 200 : 0;
        for (size_t i = start; i < lossHistory.size(); ++i) {
            if (i > start) oss << ",";
            oss << std::fixed << std::setprecision(6) << lossHistory[i];
        }
        oss << "]\n}";
        
        return oss.str();
    }
    
    // Getters
    int getEpoch() const { return epoch; }
    const std::vector<double>& getLossHistory() const { return lossHistory; }
};

} // namespace core
} // namespace neuraltrainer

#endif // NEURALTRAINER_NETWORK_H
