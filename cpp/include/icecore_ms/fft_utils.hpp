#ifndef ICECORE_MS_FFT_UTILS_HPP
#define ICECORE_MS_FFT_UTILS_HPP

#include <complex>
#include <cstddef>
#include <vector>

namespace icecore_ms {

using Complex = std::complex<double>;

std::size_t next_power_of_two(std::size_t n);

std::vector<Complex> fft(const std::vector<double>& input);

std::vector<Complex> fft(const std::vector<Complex>& input);

std::vector<double> ifft(const std::vector<Complex>& input);

void fft_inplace(std::vector<Complex>& data, bool inverse);

std::vector<double> compute_power_spectrum(const std::vector<double>& signal);

std::vector<double> bandpass_filter(
    const std::vector<double>& signal,
    double low_cutoff,
    double high_cutoff,
    double sampling_rate
);

std::vector<double> hanning_window(std::size_t size);

std::vector<double> stft(
    const std::vector<double>& signal,
    std::size_t window_size,
    std::size_t hop_size
);

std::vector<double> istft(
    const std::vector<double>& stft_matrix,
    std::size_t window_size,
    std::size_t hop_size,
    std::size_t original_length
);

}  // namespace icecore_ms

#endif  // ICECORE_MS_FFT_UTILS_HPP
