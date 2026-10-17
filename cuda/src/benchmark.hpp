#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace perf_scope {

struct Options {
    std::string_view workload;
    std::string_view variant;
    std::vector<int> shape;
    std::uint64_t seed;
    int warmups;
    int iterations;
};

struct Measurement {
    double latency_ms;
    double logical_bytes;
    bool correct;
    double flops{};
    int cublas_version{};
};

inline void cuda_check(cudaError_t status, std::string_view operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

class DeviceArray {
public:
    explicit DeviceArray(std::size_t count) {
        cuda_check(cudaMalloc(&data_, count * sizeof(float)), "allocate device array");
    }
    ~DeviceArray() { cudaFree(data_); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    float* data() const { return data_; }

private:
    float* data_{};
};

class Event {
public:
    Event() { cuda_check(cudaEventCreate(&event_), "create event"); }
    ~Event() { cudaEventDestroy(event_); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;

    cudaEvent_t native() const { return event_; }

private:
    cudaEvent_t event_{};
};

template <class Launch>
double measure(const Options& options, const Launch& launch) {
    Event start, stop;
    for (int i = 0; i < options.warmups; ++i) {
        launch();
    }
    cuda_check(cudaGetLastError(), "launch warmup");
    cuda_check(cudaDeviceSynchronize(), "synchronize warmup");
    cuda_check(cudaEventRecord(start.native()), "record start");
    for (int i = 0; i < options.iterations; ++i) {
        launch();
    }
    cuda_check(cudaGetLastError(), "launch benchmark");
    cuda_check(cudaEventRecord(stop.native()), "record stop");
    cuda_check(cudaEventSynchronize(stop.native()), "synchronize benchmark");
    float elapsed = 0;
    cuda_check(cudaEventElapsedTime(&elapsed, start.native(), stop.native()), "read timing");
    if (!(elapsed > 0) || !std::isfinite(elapsed)) {
        throw std::runtime_error("CUDA events did not resolve a positive block duration");
    }
    return static_cast<double>(elapsed) / options.iterations;
}

inline std::vector<float> input_values(std::size_t count, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<float> values(count);
    for (float& value : values) {
        value = static_cast<float>(rng() & 0xffff) / 32768.0f - 1.0f;
    }
    return values;
}

Measurement transpose(const Options& options, const cudaDeviceProp& properties);
Measurement gemm(const Options& options, const cudaDeviceProp& properties);
Measurement triad(const Options& options, const cudaDeviceProp& properties);
Measurement softmax(const Options& options, const cudaDeviceProp& properties);

}
