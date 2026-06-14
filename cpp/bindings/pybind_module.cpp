#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <vector>

#include "icecore_ms/kalman_filter.hpp"
#include "icecore_ms/fft_utils.hpp"
#include "icecore_ms/signal_utils.hpp"

namespace py = pybind11;

namespace {

std::vector<double> numpy_to_vector(const py::array_t<double>& arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Expected 1-dimensional numpy array");
    }
    const double* ptr = static_cast<const double*>(buf.ptr);
    return std::vector<double>(ptr, ptr + buf.shape[0]);
}

py::array_t<double> vector_to_numpy(const std::vector<double>& vec) {
    py::array_t<double> result(vec.size());
    py::buffer_info buf = result.request();
    double* ptr = static_cast<double*>(buf.ptr);
    std::copy(vec.begin(), vec.end(), ptr);
    return result;
}

py::array_t<bool> vector_bool_to_numpy(const std::vector<bool>& vec) {
    py::array_t<bool> result(vec.size());
    py::buffer_info buf = result.request();
    bool* ptr = static_cast<bool*>(buf.ptr);
    for (std::size_t i = 0; i < vec.size(); ++i) {
        ptr[i] = vec[i];
    }
    return result;
}

py::array_t<std::size_t> vector_size_t_to_numpy(const std::vector<std::size_t>& vec) {
    py::array_t<std::size_t> result(vec.size());
    py::buffer_info buf = result.request();
    std::size_t* ptr = static_cast<std::size_t*>(buf.ptr);
    std::copy(vec.begin(), vec.end(), ptr);
    return result;
}

}  // namespace

PYBIND11_MODULE(_icecore_native, m) {
    m.doc() = "Ice core mass spectrometry native signal processing extension";

    py::class_<icecore_ms::AdaptiveKalmanConfig>(m, "AdaptiveKalmanConfig")
        .def(py::init<>())
        .def_readwrite("process_noise_init", &icecore_ms::AdaptiveKalmanConfig::process_noise_init)
        .def_readwrite("measurement_noise_init", &icecore_ms::AdaptiveKalmanConfig::measurement_noise_init)
        .def_readwrite("adaptation_rate", &icecore_ms::AdaptiveKalmanConfig::adaptation_rate)
        .def_readwrite("innovation_momentum", &icecore_ms::AdaptiveKalmanConfig::innovation_momentum)
        .def_readwrite("outlier_threshold_sigma", &icecore_ms::AdaptiveKalmanConfig::outlier_threshold_sigma)
        .def_readwrite("enable_frequency_domain", &icecore_ms::AdaptiveKalmanConfig::enable_frequency_domain)
        .def_readwrite("fft_window", &icecore_ms::AdaptiveKalmanConfig::fft_window)
        .def_readwrite("fft_overlap", &icecore_ms::AdaptiveKalmanConfig::fft_overlap)
        .def_readwrite("low_cutoff_hz", &icecore_ms::AdaptiveKalmanConfig::low_cutoff_hz)
        .def_readwrite("high_cutoff_hz", &icecore_ms::AdaptiveKalmanConfig::high_cutoff_hz)
        .def_readwrite("sampling_rate_hz", &icecore_ms::AdaptiveKalmanConfig::sampling_rate_hz);

    py::class_<icecore_ms::AdaptiveKalmanState>(m, "AdaptiveKalmanState")
        .def_property_readonly("state_estimate",
            [](const icecore_ms::AdaptiveKalmanState& s) {
                return vector_to_numpy(s.state_estimate);
            })
        .def_property_readonly("error_covariance",
            [](const icecore_ms::AdaptiveKalmanState& s) {
                return vector_to_numpy(s.error_covariance);
            })
        .def_property_readonly("innovation_history",
            [](const icecore_ms::AdaptiveKalmanState& s) {
                return vector_to_numpy(s.innovation_history);
            })
        .def_property_readonly("process_noise_trajectory",
            [](const icecore_ms::AdaptiveKalmanState& s) {
                return vector_to_numpy(s.process_noise_trajectory);
            })
        .def_property_readonly("measurement_noise_trajectory",
            [](const icecore_ms::AdaptiveKalmanState& s) {
                return vector_to_numpy(s.measurement_noise_trajectory);
            })
        .def_readonly("rms_innovation", &icecore_ms::AdaptiveKalmanState::rms_innovation)
        .def_readonly("mean_innovation", &icecore_ms::AdaptiveKalmanState::mean_innovation);

    py::class_<icecore_ms::AdaptiveKalmanFilter1D>(m, "AdaptiveKalmanFilter1D")
        .def(py::init<const icecore_ms::AdaptiveKalmanConfig&>())
        .def("filter", [](icecore_ms::AdaptiveKalmanFilter1D& self, const py::array_t<double>& signal) {
            return vector_to_numpy(self.filter(numpy_to_vector(signal)));
        }, py::arg("signal"))
        .def("smooth", [](icecore_ms::AdaptiveKalmanFilter1D& self, const py::array_t<double>& signal) {
            return vector_to_numpy(self.smooth(numpy_to_vector(signal)));
        }, py::arg("signal"))
        .def_property_readonly("state", &icecore_ms::AdaptiveKalmanFilter1D::state)
        .def("reset", &icecore_ms::AdaptiveKalmanFilter1D::reset);

    m.def("adaptive_kalman_filter",
        [](const py::array_t<double>& signal, const icecore_ms::AdaptiveKalmanConfig& config) {
            return vector_to_numpy(icecore_ms::adaptive_kalman_filter(numpy_to_vector(signal), config));
        },
        py::arg("signal"), py::arg("config"),
        "Apply adaptive Kalman filtering to a 1D signal.");

    m.def("adaptive_kalman_smooth",
        [](const py::array_t<double>& signal, const icecore_ms::AdaptiveKalmanConfig& config) {
            return vector_to_numpy(icecore_ms::adaptive_kalman_smooth(numpy_to_vector(signal), config));
        },
        py::arg("signal"), py::arg("config"),
        "Apply bidirectional adaptive Kalman smoothing to a 1D signal.");

    m.def("detect_outliers",
        [](const py::array_t<double>& signal,
           const py::array_t<double>& filtered,
           double threshold_sigma) {
            return vector_bool_to_numpy(
                icecore_ms::detect_outliers(
                    numpy_to_vector(signal),
                    numpy_to_vector(filtered),
                    threshold_sigma
                )
            );
        },
        py::arg("signal"), py::arg("filtered"), py::arg("threshold_sigma") = 5.0,
        "Detect outliers based on residuals from filtered signal.");

    m.def("suppress_artifacts",
        [](const py::array_t<double>& signal,
           const py::array_t<bool>& outlier_mask,
           std::size_t interpolation_window) {
            py::buffer_info buf = outlier_mask.request();
            const bool* ptr = static_cast<const bool*>(buf.ptr);
            std::vector<bool> mask(ptr, ptr + buf.shape[0]);
            return vector_to_numpy(
                icecore_ms::suppress_artifacts(numpy_to_vector(signal), mask, interpolation_window)
            );
        },
        py::arg("signal"), py::arg("outlier_mask"), py::arg("interpolation_window") = 5,
        "Interpolate outlier artifacts using neighboring samples.");

    m.def("next_power_of_two", &icecore_ms::next_power_of_two,
        py::arg("n"),
        "Compute the next power of two greater than or equal to n.");

    m.def("compute_power_spectrum",
        [](const py::array_t<double>& signal) {
            return vector_to_numpy(icecore_ms::compute_power_spectrum(numpy_to_vector(signal)));
        },
        py::arg("signal"),
        "Compute the power spectrum of a real-valued signal.");

    m.def("bandpass_filter",
        [](const py::array_t<double>& signal,
           double low_cutoff, double high_cutoff, double sampling_rate) {
            return vector_to_numpy(
                icecore_ms::bandpass_filter(numpy_to_vector(signal), low_cutoff, high_cutoff, sampling_rate)
            );
        },
        py::arg("signal"), py::arg("low_cutoff"), py::arg("high_cutoff"), py::arg("sampling_rate"),
        "Apply frequency-domain bandpass filter.");

    m.def("hanning_window",
        [](std::size_t size) {
            return vector_to_numpy(icecore_ms::hanning_window(size));
        },
        py::arg("size"),
        "Generate a Hanning window of given size.");

    m.def("stft",
        [](const py::array_t<double>& signal,
           std::size_t window_size, std::size_t hop_size) {
            return vector_to_numpy(icecore_ms::stft(numpy_to_vector(signal), window_size, hop_size));
        },
        py::arg("signal"), py::arg("window_size"), py::arg("hop_size"),
        "Compute short-time Fourier transform magnitude.");

    py::class_<icecore_ms::SignalStats>(m, "SignalStats")
        .def_readonly("mean", &icecore_ms::SignalStats::mean)
        .def_readonly("variance", &icecore_ms::SignalStats::variance)
        .def_readonly("std_dev", &icecore_ms::SignalStats::std_dev)
        .def_readonly("min", &icecore_ms::SignalStats::min)
        .def_readonly("max", &icecore_ms::SignalStats::max)
        .def_readonly("rms", &icecore_ms::SignalStats::rms)
        .def_readonly("snr_db", &icecore_ms::SignalStats::snr_db);

    m.def("compute_signal_stats",
        [](const py::array_t<double>& signal) {
            return icecore_ms::compute_signal_stats(numpy_to_vector(signal));
        },
        py::arg("signal"),
        "Compute basic signal statistics.");

    m.def("detrend",
        [](const py::array_t<double>& signal) {
            return vector_to_numpy(icecore_ms::detrend(numpy_to_vector(signal)));
        },
        py::arg("signal"),
        "Remove linear trend from signal.");

    m.def("normalize",
        [](const py::array_t<double>& signal) {
            return vector_to_numpy(icecore_ms::normalize(numpy_to_vector(signal)));
        },
        py::arg("signal"),
        "Z-score normalize a signal.");

    m.def("denormalize",
        [](const py::array_t<double>& normalized, double mean, double std_dev) {
            return vector_to_numpy(icecore_ms::denormalize(numpy_to_vector(normalized), mean, std_dev));
        },
        py::arg("normalized"), py::arg("original_mean"), py::arg("original_std"),
        "Reverse z-score normalization.");

    m.def("median_filter",
        [](const py::array_t<double>& signal, std::size_t kernel_size) {
            return vector_to_numpy(icecore_ms::median_filter(numpy_to_vector(signal), kernel_size));
        },
        py::arg("signal"), py::arg("kernel_size") = 5,
        "Apply running median filter.");

    m.def("moving_average",
        [](const py::array_t<double>& signal, std::size_t window_size) {
            return vector_to_numpy(icecore_ms::moving_average(numpy_to_vector(signal), window_size));
        },
        py::arg("signal"), py::arg("window_size") = 5,
        "Apply moving average filter.");

    m.def("derivative",
        [](const py::array_t<double>& signal) {
            return vector_to_numpy(icecore_ms::derivative(numpy_to_vector(signal)));
        },
        py::arg("signal"),
        "Compute numerical first derivative.");

    m.def("find_peaks",
        [](const py::array_t<double>& signal, double min_height, double min_distance) {
            return vector_size_t_to_numpy(
                icecore_ms::find_peaks(numpy_to_vector(signal), min_height, min_distance)
            );
        },
        py::arg("signal"), py::arg("min_height"), py::arg("min_distance") = 1,
        "Find indices of local maxima in the signal.");
}
