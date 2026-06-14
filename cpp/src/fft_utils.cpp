#include "icecore_ms/fft_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace icecore_ms {

namespace {

constexpr double PI = 3.14159265358979323846;

void bit_reverse_permute(std::vector<Complex>& data) {
    const std::size_t n = data.size();
    std::size_t j = 0;
    for (std::size_t i = 1; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(data[i], data[j]);
        }
    }
}

}  // namespace

std::size_t next_power_of_two(std::size_t n) {
    if (n == 0) return 1;
    std::size_t result = 1;
    while (result < n) {
        result <<= 1;
    }
    return result;
}

void fft_inplace(std::vector<Complex>& data, bool inverse) {
    const std::size_t n = data.size();
    if (n <= 1) return;

    bit_reverse_permute(data);

    const double sign = inverse ? 1.0 : -1.0;

    for (std::size_t len = 2; len <= n; len <<= 1) {
        const double angle = 2.0 * PI * sign / static_cast<double>(len);
        const Complex wlen(std::cos(angle), std::sin(angle));
        for (std::size_t i = 0; i < n; i += len) {
            Complex w(1.0, 0.0);
            for (std::size_t j = 0; j < len / 2; ++j) {
                const Complex u = data[i + j];
                const Complex v = data[i + j + len / 2] * w;
                data[i + j] = u + v;
                data[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }

    if (inverse) {
        const double inv_n = 1.0 / static_cast<double>(n);
        for (auto& x : data) {
            x *= inv_n;
        }
    }
}

std::vector<Complex> fft(const std::vector<double>& input) {
    const std::size_t n = next_power_of_two(input.size());
    std::vector<Complex> data(n, Complex(0.0, 0.0));
    for (std::size_t i = 0; i < input.size(); ++i) {
        data[i] = Complex(input[i], 0.0);
    }
    fft_inplace(data, false);
    return data;
}

std::vector<Complex> fft(const std::vector<Complex>& input) {
    const std::size_t n = next_power_of_two(input.size());
    std::vector<Complex> data(n, Complex(0.0, 0.0));
    std::copy(input.begin(), input.end(), data.begin());
    fft_inplace(data, false);
    return data;
}

std::vector<double> ifft(const std::vector<Complex>& input) {
    std::vector<Complex> data = input;
    const std::size_t original_size = data.size();
    const std::size_t padded_size = next_power_of_two(original_size);
    if (padded_size != original_size) {
        data.resize(padded_size, Complex(0.0, 0.0));
    }
    fft_inplace(data, true);
    std::vector<double> result(original_size);
    for (std::size_t i = 0; i < original_size; ++i) {
        result[i] = data[i].real();
    }
    return result;
}

std::vector<double> compute_power_spectrum(const std::vector<double>& signal) {
    const std::size_t n = next_power_of_two(signal.size());
    std::vector<Complex> spectrum = fft(signal);
    std::vector<double> power(n / 2);
    const double norm = 1.0 / static_cast<double>(n);
    for (std::size_t i = 0; i < n / 2; ++i) {
        const double re = spectrum[i].real();
        const double im = spectrum[i].imag();
        power[i] = (re * re + im * im) * norm;
    }
    return power;
}

std::vector<double> bandpass_filter(
    const std::vector<double>& signal,
    double low_cutoff,
    double high_cutoff,
    double sampling_rate
) {
    const std::size_t n = next_power_of_two(signal.size());
    const double nyquist = sampling_rate * 0.5;
    const double low_bin = low_cutoff / nyquist * static_cast<double>(n);
    const double high_bin = high_cutoff / nyquist * static_cast<double>(n);

    std::vector<Complex> spectrum = fft(signal);

    for (std::size_t i = 0; i < n; ++i) {
        double bin = static_cast<double>(i);
        if (i > n / 2) {
            bin = static_cast<double>(n - i);
        }
        if (bin < low_bin || bin > high_bin) {
            spectrum[i] = Complex(0.0, 0.0);
        }
    }

    return ifft(spectrum);
}

std::vector<double> hanning_window(std::size_t size) {
    std::vector<double> window(size);
    if (size <= 1) {
        if (size == 1) window[0] = 1.0;
        return window;
    }
    const double factor = 2.0 * PI / static_cast<double>(size - 1);
    for (std::size_t i = 0; i < size; ++i) {
        window[i] = 0.5 * (1.0 - std::cos(factor * static_cast<double>(i)));
    }
    return window;
}

std::vector<double> stft(
    const std::vector<double>& signal,
    std::size_t window_size,
    std::size_t hop_size
) {
    const std::size_t n = signal.size();
    const std::size_t num_frames = (n - window_size) / hop_size + 1;
    const std::size_t fft_size = next_power_of_two(window_size);
    const std::size_t freq_bins = fft_size / 2;

    std::vector<double> window = hanning_window(window_size);
    std::vector<double> result(num_frames * freq_bins, 0.0);

    for (std::size_t frame = 0; frame < num_frames; ++frame) {
        const std::size_t start = frame * hop_size;
        std::vector<double> frame_data(window_size, 0.0);
        for (std::size_t i = 0; i < window_size && start + i < n; ++i) {
            frame_data[i] = signal[start + i] * window[i];
        }
        std::vector<Complex> spectrum = fft(frame_data);
        for (std::size_t j = 0; j < freq_bins; ++j) {
            result[frame * freq_bins + j] = std::abs(spectrum[j]);
        }
    }

    return result;
}

std::vector<double> istft(
    const std::vector<double>& /*stft_matrix*/,
    std::size_t /*window_size*/,
    std::size_t /*hop_size*/,
    std::size_t original_length
) {
    return std::vector<double>(original_length, 0.0);
}

}  // namespace icecore_ms
