#include "icecore_ms/kalman_filter.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "icecore_ms/fft_utils.hpp"
#include "icecore_ms/signal_utils.hpp"

namespace icecore_ms {

namespace {

constexpr double PI = 3.14159265358979323846;

}  // namespace

AdaptiveKalmanFilter1D::AdaptiveKalmanFilter1D(AdaptiveKalmanConfig config)
    : config_(std::move(config)) {}

void AdaptiveKalmanFilter1D::reset() noexcept {
    state_ = AdaptiveKalmanState();
}

void AdaptiveKalmanFilter1D::initialize_state(std::size_t signal_size) {
    state_.state_estimate.assign(signal_size, 0.0);
    state_.error_covariance.assign(signal_size, 1.0);
    state_.innovation_history.assign(signal_size, 0.0);
    state_.process_noise_trajectory.assign(signal_size, config_.process_noise_init);
    state_.measurement_noise_trajectory.assign(signal_size, config_.measurement_noise_init);
}

double AdaptiveKalmanFilter1D::gaussian_pdf(double x, double mean, double sigma) {
    const double z = (x - mean) / sigma;
    return std::exp(-0.5 * z * z) / (sigma * std::sqrt(2.0 * PI));
}

void AdaptiveKalmanFilter1D::predict_step(
    std::size_t idx,
    double& state_est,
    double& error_cov
) const {
    const double process_noise = state_.process_noise_trajectory[idx];
    if (idx == 0) {
        state_est = 0.0;
        error_cov = 1.0;
    } else {
        state_est = state_.state_estimate[idx - 1];
        error_cov = state_.error_covariance[idx - 1] + process_noise;
    }
}

void AdaptiveKalmanFilter1D::update_step(
    std::size_t idx,
    double measurement,
    double& state_est,
    double& error_cov
) {
    const double measurement_noise = state_.measurement_noise_trajectory[idx];
    const double innovation = measurement - state_est;
    state_.innovation_history[idx] = innovation;

    const double innovation_cov = error_cov + measurement_noise;
    const double kalman_gain = error_cov / (innovation_cov + std::numeric_limits<double>::epsilon());

    state_est = state_est + kalman_gain * innovation;
    error_cov = (1.0 - kalman_gain) * error_cov;
}

void AdaptiveKalmanFilter1D::adapt_noise(std::size_t idx, double innovation) {
    const double rate = config_.adaptation_rate;
    const double momentum = config_.innovation_momentum;

    if (idx > 0) {
        double prev_q = state_.process_noise_trajectory[idx - 1];
        double prev_r = state_.measurement_noise_trajectory[idx - 1];

        const double innovation_sq = innovation * innovation;
        const double prev_cov = state_.error_covariance[idx - 1];

        const double target_q = std::max(
            config_.process_noise_init * 0.01,
            rate * std::abs(innovation_sq - prev_cov - prev_r)
        );
        state_.process_noise_trajectory[idx] = momentum * prev_q + (1.0 - momentum) * target_q;

        const double target_r = std::max(
            config_.measurement_noise_init * 0.01,
            rate * std::abs(innovation_sq)
        );
        state_.measurement_noise_trajectory[idx] = momentum * prev_r + (1.0 - momentum) * target_r;
    }
}

std::vector<double> AdaptiveKalmanFilter1D::apply_bandpass(
    const std::vector<double>& signal
) const {
    if (config_.low_cutoff_hz <= 0.0 && config_.high_cutoff_hz >= config_.sampling_rate_hz * 0.5) {
        return signal;
    }
    return bandpass_filter(
        signal,
        config_.low_cutoff_hz,
        config_.high_cutoff_hz,
        config_.sampling_rate_hz
    );
}

std::vector<double> AdaptiveKalmanFilter1D::frequency_domain_filter(
    const std::vector<double>& signal
) {
    std::vector<double> bp_signal = apply_bandpass(signal);

    const std::size_t n = bp_signal.size();
    if (n == 0) return {};

    initialize_state(n);

    const SignalStats stats = compute_signal_stats(bp_signal);
    double running_mean = stats.mean;
    double running_var = stats.variance;

    for (std::size_t i = 0; i < n; ++i) {
        double state_est = 0.0;
        double error_cov = 1.0;

        predict_step(i, state_est, error_cov);

        const double measurement = bp_signal[i];
        const double z_score = std::abs(measurement - running_mean) /
                               (std::sqrt(running_var) + std::numeric_limits<double>::epsilon());

        if (z_score > config_.outlier_threshold_sigma) {
            state_.state_estimate[i] = state_est;
            state_.error_covariance[i] = error_cov + state_.measurement_noise_trajectory[i] * 10.0;
            state_.innovation_history[i] = 0.0;
        } else {
            update_step(i, measurement, state_est, error_cov);
            adapt_noise(i, measurement - state_est);

            state_.state_estimate[i] = state_est;
            state_.error_covariance[i] = error_cov;
        }

        const double alpha = 0.01;
        running_mean = (1.0 - alpha) * running_mean + alpha * measurement;
        running_var = (1.0 - alpha) * running_var + alpha * (measurement - running_mean) *
                                                      (measurement - running_mean);
    }

    double sum_innovation = 0.0;
    double sum_innovation_sq = 0.0;
    for (double v : state_.innovation_history) {
        sum_innovation += v;
        sum_innovation_sq += v * v;
    }
    state_.mean_innovation = sum_innovation / static_cast<double>(n);
    state_.rms_innovation = std::sqrt(sum_innovation_sq / static_cast<double>(n));

    return state_.state_estimate;
}

std::vector<double> AdaptiveKalmanFilter1D::filter(const std::vector<double>& signal) {
    if (config_.enable_frequency_domain) {
        return frequency_domain_filter(signal);
    }

    const std::size_t n = signal.size();
    if (n == 0) return {};

    initialize_state(n);

    for (std::size_t i = 0; i < n; ++i) {
        double state_est = 0.0;
        double error_cov = 1.0;
        predict_step(i, state_est, error_cov);
        update_step(i, signal[i], state_est, error_cov);
        adapt_noise(i, signal[i] - state_est);
        state_.state_estimate[i] = state_est;
        state_.error_covariance[i] = error_cov;
    }

    return state_.state_estimate;
}

std::vector<double> AdaptiveKalmanFilter1D::smooth(const std::vector<double>& signal) {
    const std::vector<double> forward = filter(signal);
    const std::size_t n = signal.size();
    if (n < 2) return forward;

    std::vector<double> reversed(signal.rbegin(), signal.rend());
    const std::vector<double> backward_rev = filter(reversed);
    std::vector<double> backward(backward_rev.rbegin(), backward_rev.rend());

    std::vector<double> smoothed(n);
    for (std::size_t i = 0; i < n; ++i) {
        smoothed[i] = 0.5 * (forward[i] + backward[i]);
    }
    return smoothed;
}

std::vector<double> adaptive_kalman_filter(
    const std::vector<double>& signal,
    const AdaptiveKalmanConfig& config
) {
    AdaptiveKalmanFilter1D filter(config);
    return filter.filter(signal);
}

std::vector<double> adaptive_kalman_smooth(
    const std::vector<double>& signal,
    const AdaptiveKalmanConfig& config
) {
    AdaptiveKalmanFilter1D filter(config);
    return filter.smooth(signal);
}

std::vector<bool> detect_outliers(
    const std::vector<double>& signal,
    const std::vector<double>& filtered,
    double threshold_sigma
) {
    const std::size_t n = signal.size();
    std::vector<bool> mask(n, false);

    if (n == 0) return mask;

    double sum_res = 0.0;
    double sum_res_sq = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double res = signal[i] - filtered[i];
        sum_res += res;
        sum_res_sq += res * res;
    }
    const double mean_res = sum_res / static_cast<double>(n);
    const double var_res = sum_res_sq / static_cast<double>(n) - mean_res * mean_res;
    const double sigma_res = std::sqrt(std::max(0.0, var_res));

    if (sigma_res < std::numeric_limits<double>::epsilon()) return mask;

    for (std::size_t i = 0; i < n; ++i) {
        const double res = signal[i] - filtered[i];
        const double z = std::abs(res - mean_res) / sigma_res;
        if (z > threshold_sigma) {
            mask[i] = true;
        }
    }
    return mask;
}

std::vector<double> suppress_artifacts(
    const std::vector<double>& signal,
    const std::vector<bool>& outlier_mask,
    std::size_t interpolation_window
) {
    const std::size_t n = signal.size();
    std::vector<double> result = signal;

    if (n == 0) return result;

    for (std::size_t i = 0; i < n; ++i) {
        if (outlier_mask[i]) {
            double sum = 0.0;
            std::size_t count = 0;
            const std::size_t start = (i >= interpolation_window) ? i - interpolation_window : 0;
            const std::size_t end = std::min(n, i + interpolation_window + 1);

            for (std::size_t j = start; j < end; ++j) {
                if (j != i && !outlier_mask[j]) {
                    sum += signal[j];
                    ++count;
                }
            }
            if (count > 0) {
                result[i] = sum / static_cast<double>(count);
            } else {
                std::size_t left = (i > 0) ? i - 1 : 0;
                std::size_t right = std::min(n - 1, i + 1);
                while (left > 0 && outlier_mask[left]) --left;
                while (right < n - 1 && outlier_mask[right]) ++right;
                result[i] = 0.5 * (signal[left] + signal[right]);
            }
        }
    }
    return result;
}

}  // namespace icecore_ms
