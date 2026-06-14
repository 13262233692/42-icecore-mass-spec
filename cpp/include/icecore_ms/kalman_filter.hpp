#ifndef ICECORE_MS_KALMAN_FILTER_HPP
#define ICECORE_MS_KALMAN_FILTER_HPP

#include <complex>
#include <cstddef>
#include <vector>

namespace icecore_ms {

struct AdaptiveKalmanConfig {
    double process_noise_init = 1e-4;
    double measurement_noise_init = 1e-2;
    double adaptation_rate = 0.05;
    double innovation_momentum = 0.9;
    double outlier_threshold_sigma = 5.0;
    bool enable_frequency_domain = true;
    std::size_t fft_window = 1024;
    std::size_t fft_overlap = 512;
    double low_cutoff_hz = 0.0;
    double high_cutoff_hz = 0.5;
    double sampling_rate_hz = 1.0;
};

struct AdaptiveKalmanState {
    std::vector<double> state_estimate;
    std::vector<double> error_covariance;
    std::vector<double> innovation_history;
    std::vector<double> process_noise_trajectory;
    std::vector<double> measurement_noise_trajectory;
    double rms_innovation = 0.0;
    double mean_innovation = 0.0;
};

class AdaptiveKalmanFilter1D {
public:
    explicit AdaptiveKalmanFilter1D(AdaptiveKalmanConfig config);

    std::vector<double> filter(const std::vector<double>& signal);

    std::vector<double> smooth(const std::vector<double>& signal);

    const AdaptiveKalmanState& state() const noexcept { return state_; }

    void reset() noexcept;

private:
    AdaptiveKalmanConfig config_;
    AdaptiveKalmanState state_;

    void initialize_state(std::size_t signal_size);

    void predict_step(
        std::size_t idx,
        double& state_est,
        double& error_cov
    ) const;

    void update_step(
        std::size_t idx,
        double measurement,
        double& state_est,
        double& error_cov
    );

    void adapt_noise(std::size_t idx, double innovation);

    std::vector<double> apply_bandpass(const std::vector<double>& signal) const;

    std::vector<double> frequency_domain_filter(const std::vector<double>& signal);

    static double gaussian_pdf(double x, double mean, double sigma);
};

std::vector<double> adaptive_kalman_filter(
    const std::vector<double>& signal,
    const AdaptiveKalmanConfig& config
);

std::vector<double> adaptive_kalman_smooth(
    const std::vector<double>& signal,
    const AdaptiveKalmanConfig& config
);

std::vector<bool> detect_outliers(
    const std::vector<double>& signal,
    const std::vector<double>& filtered,
    double threshold_sigma
);

std::vector<double> suppress_artifacts(
    const std::vector<double>& signal,
    const std::vector<bool>& outlier_mask,
    std::size_t interpolation_window = 5
);

}  // namespace icecore_ms

#endif  // ICECORE_MS_KALMAN_FILTER_HPP
