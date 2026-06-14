#ifndef ICECORE_MS_SIGNAL_UTILS_HPP
#define ICECORE_MS_SIGNAL_UTILS_HPP

#include <cstddef>
#include <vector>

namespace icecore_ms {

struct SignalStats {
    double mean = 0.0;
    double variance = 0.0;
    double std_dev = 0.0;
    double min = 0.0;
    double max = 0.0;
    double rms = 0.0;
    double snr_db = 0.0;
};

SignalStats compute_signal_stats(const std::vector<double>& signal);

std::vector<double> detrend(const std::vector<double>& signal);

std::vector<double> normalize(const std::vector<double>& signal);

std::vector<double> denormalize(
    const std::vector<double>& normalized,
    double original_mean,
    double original_std
);

std::vector<double> interpolate_nans(
    const std::vector<double>& signal,
    std::size_t window_size = 5
);

std::vector<double> median_filter(
    const std::vector<double>& signal,
    std::size_t kernel_size = 5
);

std::vector<double> moving_average(
    const std::vector<double>& signal,
    std::size_t window_size = 5
);

std::vector<double> derivative(const std::vector<double>& signal);

std::vector<double> convolve(
    const std::vector<double>& signal,
    const std::vector<double>& kernel
);

std::size_t find_peak_index(const std::vector<double>& signal);

std::vector<std::size_t> find_peaks(
    const std::vector<double>& signal,
    double min_height,
    double min_distance = 1
);

}  // namespace icecore_ms

#endif  // ICECORE_MS_SIGNAL_UTILS_HPP
