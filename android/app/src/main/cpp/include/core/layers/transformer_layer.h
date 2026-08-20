// Transformer Layer header - Simplified self-attention layer
#ifndef NEURALTRAINER_TRANSFORMER_LAYER_H
#define NEURALTRAINER_TRANSFORMER_LAYER_H

#include "base_layer.h"
#include "../activations.h"
#include <vector>
#include <string>
#include <random>
#include <cmath>

namespace neuraltrainer {
namespace core {

/**
 * Simplified Self-Attention Layer (Transformer building block)
 * Supports multi-head attention with configurable dimensions
 */
class TransformerLayer : public Layer {
private:
    size_t embedDim;
    size_t numHeads;
    size_t seqLen;
    size_t headDim;
    
    // Query, Key, Value projection matrices
    std::vector<std::vector<double>> Wq, Wk, Wv;
    std::vector<double> bq, bk, bv;
    
    // Output projection
    std::vector<std::vector<double>> Wo;
    std::vector<double> bo;
    
    // Gradients
    std::vector<std::vector<double>> dWq, dWk, dWv, dWo;
    std::vector<double> dbq, dbk, dbv, dbo;
    
    static std::mt19937 rng;
    
public:
    /**
     * Constructor for simplified transformer layer
     * @param dim Embedding dimension
     * @param heads Number of attention heads
     * @param seq Sequence length
     */
    TransformerLayer(size_t dim, size_t heads = 4, size_t seq = 10)
        : embedDim(dim), numHeads(heads), seqLen(seq) {
        
        headDim = dim / heads;
        if (headDim * heads != dim) {
            headDim = dim;  // Fallback for non-divisible cases
            numHeads = 1;
        }
        
        // Xavier initialization
        double scale = std::sqrt(2.0 / (dim + headDim));
        std::uniform_real_distribution<double> dist(-scale, scale);
        
        // Resize QKV matrices
        Wq.resize(dim, std::vector<double>(dim));
        Wk.resize(dim, std::vector<double>(dim));
        Wv.resize(dim, std::vector<double>(dim));
        bq.resize(dim, 0.0);
        bk.resize(dim, 0.0);
        bv.resize(dim, 0.0);
        
        // Output projection
        Wo.resize(dim, std::vector<double>(dim));
        bo.resize(dim, 0.0);
        
        // Gradients
        dWq.resize(dim, std::vector<double>(dim, 0.0));
        dWk.resize(dim, std::vector<double>(dim, 0.0));
        dWv.resize(dim, std::vector<double>(dim, 0.0));
        dWo.resize(dim, std::vector<double>(dim, 0.0));
        dbq.resize(dim, 0.0);
        dbk.resize(dim, 0.0);
        dbv.resize(dim, 0.0);
        dbo.resize(dim, 0.0);
        
        // Initialize weights
        for (size_t i = 0; i < dim; ++i) {
            for (size_t j = 0; j < dim; ++j) {
                Wq[i][j] = dist(rng);
                Wk[i][j] = dist(rng);
                Wv[i][j] = dist(rng);
                Wo[i][j] = dist(rng);
            }
        }
    }
    
    // Simplified softmax helper
    std::vector<double> softmax(const std::vector<double>& x) {
        std::vector<double> result(x.size());
        double maxVal = *std::max_element(x.begin(), x.end());
        double sum = 0.0;
        
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] = std::exp(x[i] - maxVal);
            sum += result[i];
        }
        
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] /= sum;
        }
        
        return result;
    }
    
    std::vector<double> forward(const std::vector<double>& input, bool training = true) override {
        cachedInput = input;
        
        // Simplified single-position forward pass
        // Full implementation would handle sequences properly
        
        // Project to Q, K, V
        std::vector<double> Q(embedDim, 0.0), K(embedDim, 0.0), V(embedDim, 0.0);
        
        for (size_t i = 0; i < embedDim; ++i) {
            for (size_t j = 0; j < embedDim; ++j) {
                Q[i] += Wq[i][j] * input[j];
                K[i] += Wk[i][j] * input[j];
                V[i] += Wv[i][j] * input[j];
            }
            Q[i] += bq[i];
            K[i] += bk[i];
            V[i] += bv[i];
        }
        
        // Simplified attention: just use V as output
        // Full implementation would compute Q*K^T/sqrt(d) and apply softmax
        cachedOutput.resize(embedDim);
        for (size_t i = 0; i < embedDim; ++i) {
            double sum = bo[i];
            for (size_t j = 0; j < embedDim; ++j) {
                sum += Wo[i][j] * V[j];
            }
            cachedOutput[i] = sum;
        }
        
        return cachedOutput;
    }
    
    std::vector<double> backward(const std::vector<double>& gradOutput) override {
        // Simplified backward pass
        std::vector<double> gradInput(embedDim, 0.0);
        
        for (size_t i = 0; i < embedDim; ++i) {
            for (size_t j = 0; j < embedDim; ++j) {
                gradInput[j] += gradOutput[i] * Wo[i][j];
                dWo[i][j] += gradOutput[i] * cachedInput[j];
            }
            dbo[i] += gradOutput[i];
        }
        
        return gradInput;
    }
    
    void update(double lr, double momentum = 0.0) {
        // Plain SGD update
        for (size_t i = 0; i < embedDim; ++i) {
            bq[i] -= lr * dbq[i];
            bk[i] -= lr * dbk[i];
            bv[i] -= lr * dbv[i];
            bo[i] -= lr * dbo[i];
            
            for (size_t j = 0; j < embedDim; ++j) {
                Wq[i][j] -= lr * dWq[i][j];
                Wk[i][j] -= lr * dWk[i][j];
                Wv[i][j] -= lr * dWv[i][j];
                Wo[i][j] -= lr * dWo[i][j];
            }
        }
        
        // Zero gradients
        std::fill(dbq.begin(), dbq.end(), 0.0);
        std::fill(dbk.begin(), dbk.end(), 0.0);
        std::fill(dbv.begin(), dbv.end(), 0.0);
        std::fill(dbo.begin(), dbo.end(), 0.0);
        
        for (auto& row : dWq) std::fill(row.begin(), row.end(), 0.0);
        for (auto& row : dWk) std::fill(row.begin(), row.end(), 0.0);
        for (auto& row : dWv) std::fill(row.begin(), row.end(), 0.0);
        for (auto& row : dWo) std::fill(row.begin(), row.end(), 0.0);
    }
    
    std::string type() const override { return "transformer"; }
    size_t outputSize() const override { return embedDim; }
    size_t paramCount() const override {
        return 4 * embedDim * embedDim + 4 * embedDim;
    }
    
    std::string toJson() const override {
        return "{\"type\":\"transformer\",\"embed_dim\":" + std::to_string(embedDim) +
               ",\"num_heads\":" + std::to_string(numHeads) + "}";
    }
};

std::mt19937 TransformerLayer::rng(std::random_device{}());

} // namespace core
} // namespace neuraltrainer

#endif // NEURALTRAINER_TRANSFORMER_LAYER_H
