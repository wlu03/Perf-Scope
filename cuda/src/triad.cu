#include "benchmark.hpp"

namespace perf_scope {
namespace {

constexpr float coefficient = 0.75f;
constexpr int block_width = 256;

__global__ void stream_triad(const float* a, const float* b, float* out, int count) {
    const unsigned index = blockIdx.x * block_width + threadIdx.x;
    if (index < static_cast<unsigned>(count)) {
        out[index] = fmaf(coefficient, b[index], a[index]);
    }
}

}

Measurement triad(const Options& options, const cudaDeviceProp& properties) {
    const int count = options.shape[0];
    const unsigned blocks = (count - 1) / block_width + 1;
    if (blocks > static_cast<unsigned>(properties.maxGridSize[0])) {
        throw std::invalid_argument("triad shape exceeds CUDA grid dimensions");
    }
    const std::size_t bytes = static_cast<std::size_t>(count) * sizeof(float);
    DeviceArray a_gpu(count), b_gpu(count), out_gpu(count);
    const auto a = input_values(count, options.seed);
    const auto b = input_values(count, options.seed + 1);
    std::vector<float> result(count);
    cuda_check(cudaMemcpy(a_gpu.data(), a.data(), bytes, cudaMemcpyHostToDevice), "upload triad input A");
    cuda_check(cudaMemcpy(b_gpu.data(), b.data(), bytes, cudaMemcpyHostToDevice), "upload triad input B");
    const double latency = measure(options, [&] {
        stream_triad<<<blocks, block_width>>>(a_gpu.data(), b_gpu.data(), out_gpu.data(), count);
    });
    cuda_check(cudaMemcpy(result.data(), out_gpu.data(), bytes, cudaMemcpyDeviceToHost), "download triad output");
    for (int index = 0; index < count; ++index) {
        if (result[index] != std::fma(coefficient, b[index], a[index])) {
            return {latency, 3.0 * bytes, false};
        }
    }
    return {latency, 3.0 * bytes, true};
}

}
