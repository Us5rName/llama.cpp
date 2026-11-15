#pragma once

#include "speculative.h" // common_params_speculative_draft, llama_seq_id
#include "log.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

// Simple bounded ring buffer for adaptive-length history tracking copied from common/sampling.cpp.
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

// Adaptive speculative draft-length heuristic.
//
// `n_cur` is the current draft length — the number of tokens the draft model
// generates before the target model verifies them.
//
// May adjust `n_cur` by ±1 after each draft verification, bounded by `[n_min, n_max]`, based on recent
// verification outcomes. The heuristic has four interacting components:
//
// 1. Rolling window  —  A ring buffer records the last N verification results as
//    success (1) or failure (0). A step is a "success" if the number of accepted
//    tokens is within `adaptive_length_bias` of the predicted length. As the window
//    slides, the `last_successes` / `last_failures` counters are kept in sync.
//
// 2. Zones            —  The window counts determine which zone the heuristic is in:
//    - Positive zone:  `last_successes >= positive_zone`  → draft model is performing well
//    - Negative zone:  `last_failures  >= negative_zone`  → draft model is underperforming
//    - Neutral:        neither threshold met              → uncertainty / mixed signal
//
//    Only in a zone does the heuristic accumulate evidence. In the positive zone it
//    increments `correct_pred`; in the negative zone it increments `wrong_pred`. Being
//    in the neutral zone resets both counters to zero. This prevents reacting to
//    transient spikes.
//
// 3. Threshold         —  A length change fires only after `correct_pred` or `wrong_pred`
//    reaches `adaptive_length_threshold`. This requires sustained evidence over multiple
//    verification steps, not just a single good/bad round.
//
// 4. Cooldown          —  After a length change, all mutable state is reset and a
//    cooldown is started. The `heuristic_token_counter` increments by `n_cur` each
//    verification step until it reaches `heuristic_variance_limit`, at which point it
//    is set to -1. This sentinel value means the cooldown has expired and a length
//    change is now allowed. During cooldown (counter > -1), no length change can
//    occur. This prevents rapid oscillation and gives the new length enough tokens
//    to produce a stable signal before another adjustment.
//
// Flow per verification step:
//   update() → advance cooldown → slide window → classify success/failure →
//   determine zone → accumulate consecutive counter → if threshold met AND cooldown
//   expired → adjust n_cur by ±1 → reset state.
//
// Example (parameters: n_min=2, n_max=12, n_cur=4, threshold=3,
//   positive_zone=4, negative_zone=3, history_length=8, variance_limit=16, bias=1):
//
//   Step  n_cur  accepted  window              succ/fail  zone       corr/wrong       cool  action
//   ----  -----  --------  ------------------  ---------  ---------  ---------------  ----  -------
//   1       4       4      [1]                 1 / 0      neutral    0 / 0            4     cooldown
//   2       4       4      [1,1]               2 / 0      neutral    0 / 0            8     cooldown
//   3       4       4      [1,1,1]             3 / 0      neutral    0 / 0            12    cooldown
//   4       4       4      [1,1,1,1]           4 / 0      positive   1 / 0            -1    — (cooldown expired)
//   5       4       3      [1,1,1,1,1]         5 / 0      positive   2 / 0 (bias)     -1    —
//   6       4       4      [1,1,1,1,1,1]       6 / 0      positive   3 / 0            -1    n_cur → 5
//
//   At step 6 the consecutive counter reaches the threshold and cooldown is already
//   expired, so the draft length increases. All state resets; the next round of
//   evidence is gathered with n_cur = 5.
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

    // Reset mutable state after a length change.
    void reset_state() {
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

        if (last_accepted.size() >= adaptive_history_length) {
            const auto old = last_accepted.front();
            if (old == 0) {
                last_failures--;
            } else {
                last_successes--;
            }
        }

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
                    reset_state();
                }
            } else if (wrong_pred == adaptive_length_threshold) {
                if (n_cur > n_min) {
                    n_cur--;
                    reset_state();
                }
            }
        } else {
            LOG_DBG(" - seq_id %d, adaptive length allowed to change in %d tokens\n",
                    seq_id, (int)(heuristic_variance_limit - heuristic_token_counter));
        }
    }
};
