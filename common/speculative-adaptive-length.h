#pragma once

#include "speculative.h" // common_params_speculative_draft, llama_seq_id
#include "log.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

// Simple bounded ring buffer for adaptive-length history tracking.
template<typename T>
struct ring_buffer {
    ring_buffer() : capacity(0), sz(0), first(0), pos(0) {}
    explicit ring_buffer(size_t cap) : capacity(cap), sz(0), first(0), pos(0), data(cap) {}

    T & front() {
        if (sz == 0) throw std::runtime_error("ring buffer is empty");
        return data[first];
    }

    const T & front() const {
        if (sz == 0) throw std::runtime_error("ring buffer is empty");
        return data[first];
    }

    T & back() {
        if (sz == 0) throw std::runtime_error("ring buffer is empty");
        return data[pos];
    }

    const T & back() const {
        if (sz == 0) throw std::runtime_error("ring buffer is empty");
        return data[pos];
    }

    void push_back(const T & value) {
        if (sz == capacity) {
            first = (first + 1) % capacity;
        } else {
            sz++;
        }
        data[pos] = value;
        pos = (pos + 1) % capacity;
    }

    T pop_front() {
        if (sz == 0) throw std::runtime_error("ring buffer is empty");
        T value = data[first];
        first = (first + 1) % capacity;
        sz--;
        return value;
    }

    const T & rat(size_t i) const {
        if (i >= sz) throw std::runtime_error("ring buffer: index out of bounds");
        return data[(first + sz - i - 1) % capacity];
    }

    std::vector<T> to_vector() const {
        std::vector<T> result;
        result.reserve(sz);
        for (size_t i = 0; i < sz; i++) {
            result.push_back(data[(first + i) % capacity]);
        }
        return result;
    }

    void clear() {
        sz = 0;
        first = 0;
        pos = 0;
    }

    bool empty() const { return sz == 0; }
    size_t size() const { return sz; }

    size_t capacity = 0;
    size_t sz = 0;
    size_t first = 0;
    size_t pos = 0;
    std::vector<T> data;
};

// Adaptive draft-length controller for speculative decoding.
//
// All configuration is copied from common_params_speculative_draft at
// construction time.  The struct then owns both config and per-sequence
// mutable state.
struct common_speculative_adaptive_length {
    // ---- config (copied from params in the constructor) ----
    int32_t  adaptive_length_threshold;
    int32_t  adaptive_length_bias;
    size_t   adaptive_history_length;
    int32_t  heuristic_variance_limit;
    int32_t  n_min;
    int32_t  n_max;
    int32_t  adaptive_length_positive_zone;
    int32_t  adaptive_length_negative_zone;

    // ---- per-sequence mutable state ----
    int32_t  n_cur;
    int32_t  correct_pred;
    int32_t  wrong_pred;
    int32_t  heuristic_token_counter;
    ring_buffer<int> last_accepted;
    int32_t  last_successes;
    int32_t  last_failures;

    // Construct by copying all needed config from params.
    explicit common_speculative_adaptive_length(
        const common_params_speculative_draft & p)
        : adaptive_length_threshold(p.adaptive_length_threshold)
        , adaptive_length_bias        (p.adaptive_length_bias)
        , adaptive_history_length     (p.adaptive_history_length)
        , heuristic_variance_limit    (p.adaptive_length_variance_limit)
        , n_min                       (p.n_min)
        , n_max                       (p.n_max)
        , adaptive_length_positive_zone(p.adaptive_length_positive_zone)
        , adaptive_length_negative_zone(p.adaptive_length_negative_zone)
        , n_cur                       (p.n_min)
        , correct_pred                (0)
        , wrong_pred                  (0)
        , heuristic_token_counter     (0)
        , last_accepted               (p.adaptive_history_length)
        , last_successes              (0)
        , last_failures               (0)
    {}

    // Reset mutable state to initial values (keeps config intact).
    void init() {
        n_cur                 = n_min;
        correct_pred          = 0;
        wrong_pred            = 0;
        heuristic_token_counter = 0;
        last_accepted.clear();
        last_successes        = 0;
        last_failures         = 0;
    }

    // Feed one verification result.
    void update(int32_t n_accepted, int32_t seq_id) {
        if (adaptive_length_threshold <= 0) {
            return;
        }

        // ---- heuristic token cooldown ----
        if (heuristic_token_counter > -1) {
            heuristic_token_counter += n_cur;
            if (heuristic_token_counter >= heuristic_variance_limit) {
                heuristic_token_counter = -1; // cooldown done
            }
        }

        LOG_DBG(" - seq_id %d, adaptive draft predicted %d/%d\n", seq_id, n_accepted, n_cur);

        // ---- rolling window: evict oldest if full ----
        if (last_accepted.size() >= adaptive_history_length) {
            const auto old = last_accepted.front();
            if (old == 0) {
                last_failures--;
            } else {
                last_successes--;
            }
        }

        // ---- record current verification result ----
        if (n_accepted >= n_cur - adaptive_length_bias) {
            last_accepted.push_back(1);
            last_successes++;
        } else {
            last_accepted.push_back(0);
            last_failures++;
        }

        // ---- determine zone and update consecutive counters ----
        bool in_positive_zone = last_successes >= adaptive_length_positive_zone;
        bool in_negative_zone = last_failures  >= adaptive_length_negative_zone;

        if (in_positive_zone) {
            if (correct_pred < adaptive_length_threshold) {
                correct_pred++;
            }
            wrong_pred = 0;
            LOG_DBG(" - seq_id %d, adaptive conscutives successes %d\n", seq_id, correct_pred);
        } else if (in_negative_zone) {
            if (wrong_pred < adaptive_length_threshold) {
                wrong_pred++;
            }
            correct_pred = 0;
            LOG_DBG(" - seq_id %d, adaptive conscutives failures %d\n", seq_id, wrong_pred);
        } else {
            correct_pred = 0;
            wrong_pred = 0;
        }

        LOG_DBG(" - seq_id %d, adaptive successs/failures %d/%d\n",
                seq_id, last_successes, last_failures);

        // ---- apply length change if cooldown has expired ----
        if (heuristic_token_counter == -1) {
            if (correct_pred == adaptive_length_threshold) {
                if (n_cur < n_max) {
                    n_cur++;
                    init();
                }
            } else if (wrong_pred == adaptive_length_threshold) {
                if (n_cur > n_min) {
                    n_cur--;
                    init();
                }
            }
        } else {
            LOG_DBG(" - seq_id %d, adaptive length allowed to change in %d tokens\n",
                    seq_id, (int)(heuristic_variance_limit - heuristic_token_counter));
        }
    }
};
