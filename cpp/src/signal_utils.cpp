#include "icecore_ms/signal_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

namespace icecore_ms {

SignalStats compute_signal_stats(const std::vector<double>& signal) {
    SignalStats stats;
    if (signal.empty()) return stats;

    stats.min = *std::min_element(signal.begin(), signal.end());
    stats.max = *std::max_element(signal.begin(), signal.end());
    stats.mean = std::accumulate(signal.begin(), signal.end(), 0.0) /
                 static_cast<double>(signal.size());

    double sum_sq = 0.0;
    double sum_abs_sq = 0.0;
    for (double v : signal) {
        const double diff = v - stats.mean;
        sum_sq += diff * diff;
        sum_abs_sq += v * v;
    }
    stats.variance = sum_sq / static_cast<double>(signal.size());
    stats.std_dev = std::sqrt(stats.variance);
    stats.rms = std::sqrt(sum_abs_sq / static_cast<double>(signal.size()));

    if (stats.std_dev > 1e-12) {
        stats.snr_db = 20.0 * std::log10(std::abs(stats.mean) / stats.std_dev + 1e-12);
    }
    return stats;
}

std::vector<double> detrend(const std::vector<double>& signal) {
    if (signal.size() < 2) return signal;

    const std::size_t n = signal.size();
    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_xx = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(i);
        sum_x += x;
        sum_y += signal[i];
        sum_xy += x * signal[i];
        sum_xx += x * x;
    }
    const double denom = static_cast<double>(n) * sum_xx - sum_x * sum_x;
    if (std::abs(denom) < 1e-12) {
        return signal;
    }
    const double slope = (static_cast<double>(n) * sum_xy - sum_x * sum_y) / denom;
    const double intercept = (sum_y - slope * sum_x) / static_cast<double>(n);

    std::vector<double> result(n);
    for (std::size_t i = 0; i < n; ++i) {
        result[i] = signal[i] - (slope * static_cast<double>(i) + intercept);
    }
    return result;
}

std::vector<double> normalize(const std::vector<double>& signal) {
    const SignalStats stats = compute_signal_stats(signal);
    std::vector<double> result(signal.size());
    if (stats.std_dev < 1e-12) {
        for (std::size_t i = 0; i < signal.size(); ++i) {
            result[i] = signal[i] - stats.mean;
        }
    } else {
        for (std::size_t i = 0; i < signal.size(); ++i) {
            result[i] = (signal[i] - stats.mean) / stats.std_dev;
        }
    }
    return result;
}

std::vector<double> denormalize(
    const std::vector<double>& normalized,
    double original_mean,
    double original_std
) {
    std::vector<double> result(normalized.size());
    for (std::size_t i = 0; i < normalized.size(); ++i) {
        result[i] = normalized[i] * original_std + original_mean;
    }
    return result;
}

std::vector<double> interpolate_nans(
    const std::vector<double>& signal,
    std::size_t window_size
) {
    std::vector<double> result = signal;
    const std::size_t n = signal.size();

    for (std::size_t i = 0; i < n; ++i) {
        if (std::isnan(signal[i])) {
            double sum = 0.0;
            std::size_t count = 0;
            const std::size_t start = (i >= window_size) ? i - window_size : 0;
            const std::size_t end = std::min(n, i + window_size + 1);
            for (std::size_t j = start; j < end; ++j) {
                if (!std::isnan(signal[j])) {
                    sum += signal[j];
                    ++count;
                }
            }
            result[i] = (count > 0) ? sum / static_cast<double>(count) : 0.0;
        }
    }
    return result;
}

std::vector<double> median_filter(
    const std::vector<double>& signal,
    std::size_t kernel_size
) {
    const std::size_t n = signal.size();
    if (n == 0 || kernel_size == 0) return signal;
    const std::size_t half = kernel_size / 2;

    std::vector<double> result(n);
    for (std::size_t i = 0; i < n; ++i) {
        std::vector<double> window;
        window.reserve(kernel_size);
        const std::size_t start = (i >= half) ? i - half : 0;
        const std::size_t end = std::min(n, i + half + 1);
        for (std::size_t j = start; j < end; ++j) {
            window.push_back(signal[j]);
        }
        std::sort(window.begin(), window.end());
        result[i] = window[window.size() / 2];
    }
    return result;
}

std::vector<double> moving_average(
    const std::vector<double>& signal,
    std::size_t window_size
) {
    const std::size_t n = signal.size();
    if (n == 0 || window_size == 0) return signal;

    std::vector<double> result(n);
    double sum = 0.0;
    for (std::size_t i = 0; i < window_size && i < n; ++i) {
        sum += signal[i];
    }
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t effective_window = std::min(window_size, n - i);
        if (i > 0 && i + window_size <= n) {
            sum += signal[i + window_size - 1] - signal[i - 1];
        } else if (i > 0) {
            sum -= signal[i - 1];
        }
        result[i] = sum / static_cast<double>(
            std::min(window_size, i + 1) + std::min(window_size, n - i) - 1
        );
        if (i + 1 <= window_size) {
            sum = (i + 1 < n) ? sum + signal[i + 1] : sum;
        }
    }

    std::vector<double> corrected(n);
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t start = (i >= window_size / 2) ? i - window_size / 2 : 0;
        const std::size_t end = std::min(n, i + window_size / 2 + 1);
        double s = 0.0;
        for (std::size_t j = start; j < end; ++j) {
            s += signal[j];
        }
        corrected[i] = s / static_cast<double>(end - start);
    }
    return corrected;
}

std::vector<double> derivative(const std::vector<double>& signal) {
    const std::size_t n = signal.size();
    if (n < 2) return std::vector<double>(n, 0.0);

    std::vector<double> result(n);
    result[0] = signal[1] - signal[0];
    for (std::size_t i = 1; i < n - 1; ++i) {
        result[i] = (signal[i + 1] - signal[i - 1]) * 0.5;
    }
    result[n - 1] = signal[n - 1] - signal[n - 2];
    return result;
}

std::vector<double> convolve(
    const std::vector<double>& signal,
    const std::vector<double>& kernel
) {
    const std::size_t n = signal.size();
    const std::size_t m = kernel.size();
    if (n == 0 || m == 0) return {};

    const std::size_t out_size = n + m - 1;
    std::vector<double> result(out_size, 0.0);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < m; ++j) {
            result[i + j] += signal[i] * kernel[j];
        }
    }
    return result;
}

std::size_t find_peak_index(const std::vector<double>& signal) {
    if (signal.empty()) return 0;
    return static_cast<std::size_t>(
        std::distance(signal.begin(), std::max_element(signal.begin(), signal.end()))
    );
}

std::vector<std::size_t> find_peaks(
    const std::vector<double>& signal,
    double min_height,
    double min_distance
) {
    std::vector<std::size_t> peaks;
    const std::size_t n = signal.size();
    if (n < 3) return peaks;

    const std::size_t min_dist = static_cast<std::size_t>(min_distance);
    std::size_t last_peak = (min_dist > 0) ? n : 0;

    for (std::size_t i = 1; i < n - 1; ++i) {
        if (signal[i] > min_height &&
            signal[i] > signal[i - 1] &&
            signal[i] >= signal[i + 1]) {
            if (last_peak == n || i - last_peak >= min_dist) {
                peaks.push_back(i);
                last_peak = i;
            }
        }
    }
    return peaks;
}

}  // namespace icecore_ms
