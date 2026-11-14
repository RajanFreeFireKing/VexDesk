#ifndef SAMPLER_HPP
#define SAMPLER_HPP

#include <vector>
#include <random>
#include <cstddef>
#include <cstdint>
#include "gemv.hpp"

namespace streamvex {

struct SamplerConfig {
    float temperature = 0.7f;
    int top_k = 40;
    float top_p = 0.9f;
    uint32_t seed = 42;
};

class Sampler {
public:
    explicit Sampler(const SamplerConfig& config = SamplerConfig());

    /**
     * Samples the next token ID from an array of raw unnormalized logits.
     * Applies temperature scaling, Top-K truncation, Top-P (nucleus) filtering,
     * and categorical sampling.
     * 
     * @param logits Raw unnormalized logits of size vocab_size.
     * @param vocab_size Total size of the vocabulary.
     * @return Selected next token ID.
     */
    int sample(float* logits, int vocab_size);

    void set_temperature(float temp) { m_config.temperature = temp; }
    void set_top_k(int k) { m_config.top_k = k; }
    void set_top_p(float p) { m_config.top_p = p; }
    void set_seed(uint32_t seed);

private:
    SamplerConfig m_config;
    std::mt19937 m_rng;
};

} // namespace streamvex

#endif // SAMPLER_HPP
