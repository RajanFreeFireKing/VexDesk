#include "sampler.hpp"

#include <cmath>
#include <algorithm>
#include <numeric>

namespace streamvex {

Sampler::Sampler(const SamplerConfig& config)
    : m_config(config),
      m_rng(config.seed) {}

void Sampler::set_seed(uint32_t seed) {
    m_config.seed = seed;
    m_rng.seed(seed);
}

int Sampler::sample(float* logits, int vocab_size) {
    if (vocab_size <= 0) return 0;

    // 1. Greedy decoding if temperature is near 0
    if (m_config.temperature <= 1e-4f) {
        int best_idx = 0;
        float best_val = logits[0];
        for (int i = 1; i < vocab_size; ++i) {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best_idx = i;
            }
        }
        return best_idx;
    }

    // 2. Temperature scaling
    float inv_temp = 1.0f / m_config.temperature;
    for (int i = 0; i < vocab_size; ++i) {
        logits[i] *= inv_temp;
    }

    // 3. Find max logit for numerical stability
    float max_logit = logits[0];
    for (int i = 1; i < vocab_size; ++i) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    // 4. Exponentiate to obtain unnormalized probabilities
    std::vector<std::pair<float, int>> probs(vocab_size);
    float sum_exp = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        float p = std::exp(logits[i] - max_logit);
        probs[i] = {p, i};
        sum_exp += p;
    }

    // Normalize
    float inv_sum = 1.0f / (sum_exp + 1e-9f);
    for (int i = 0; i < vocab_size; ++i) {
        probs[i].first *= inv_sum;
    }

    // Sort descending by probability
    std::sort(probs.begin(), probs.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });

    // 5. Apply Top-K truncation
    int k = m_config.top_k > 0 ? std::min(m_config.top_k, vocab_size) : vocab_size;

    // 6. Apply Top-P (Nucleus) cumulative probability cutoff
    float cumulative_p = 0.0f;
    int cutoff_idx = k;
    for (int i = 0; i < k; ++i) {
        cumulative_p += probs[i].first;
        if (cumulative_p >= m_config.top_p) {
            cutoff_idx = i + 1;
            break;
        }
    }

    // Re-normalize probabilities up to cutoff
    float active_sum = 0.0f;
    for (int i = 0; i < cutoff_idx; ++i) {
        active_sum += probs[i].first;
    }
    float inv_active = 1.0f / (active_sum + 1e-9f);

    // 7. Sample categorical distribution
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(m_rng);
    float cdf = 0.0f;

    for (int i = 0; i < cutoff_idx; ++i) {
        cdf += probs[i].first * inv_active;
        if (r <= cdf || i == cutoff_idx - 1) {
            return probs[i].second;
        }
    }

    return probs[0].second;
}

} // namespace streamvex
