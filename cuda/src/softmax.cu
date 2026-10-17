#include "benchmark.hpp"

#include <math_constants.h>

#include <algorithm>

namespace perf_scope {
namespace {

constexpr int threads_per_row = 256;

template <bool maximum>
__device__ float reduce_row(float local, float* shared) {
    shared[threadIdx.x] = local;
    __syncthreads();
    for (int stride = threads_per_row / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < static_cast<unsigned>(stride)) {
            const float other = shared[threadIdx.x + stride];
            if constexpr (maximum) {
                shared[threadIdx.x] = fmaxf(shared[threadIdx.x], other);
            } else {
                shared[threadIdx.x] += other;
            }
        }
        __syncthreads();
    }
    const float result = shared[0];
    __syncthreads();
    return result;
}

__device__ float local_max(const float* input, std::size_t columns) {
    float result = -CUDART_INF_F;
    for (std::size_t column = threadIdx.x; column < columns; column += threads_per_row) {
        result = fmaxf(result, input[column]);
    }
    return result;
}

__global__ void softmax_reduce_max(const float* input, float* maxima, std::size_t columns) {
    __shared__ float scratch[threads_per_row];
    const float* row = input + blockIdx.x * columns;
    const float maximum = reduce_row<true>(local_max(row, columns), scratch);
    if (threadIdx.x == 0) {
        maxima[blockIdx.x] = maximum;
    }
}

__global__ void softmax_exp_sum(
    const float* input, const float* maxima, float* exponentials, float* sums, std::size_t columns
) {
    __shared__ float scratch[threads_per_row];
    const std::size_t offset = blockIdx.x * columns;
    const float maximum = maxima[blockIdx.x];
    float total = 0;
    for (std::size_t column = threadIdx.x; column < columns; column += threads_per_row) {
        const float value = expf(input[offset + column] - maximum);
        exponentials[offset + column] = value;
        total += value;
    }
    const float sum = reduce_row<false>(total, scratch);
    if (threadIdx.x == 0) {
        sums[blockIdx.x] = sum;
    }
}

__global__ void softmax_normalize(
    const float* exponentials, const float* sums, float* output, std::size_t columns
) {
    const std::size_t offset = blockIdx.x * columns;
    const float sum = sums[blockIdx.x];
    for (std::size_t column = threadIdx.x; column < columns; column += threads_per_row) {
        output[offset + column] = exponentials[offset + column] / sum;
    }
}

__global__ void softmax_fused(const float* input, float* output, std::size_t columns) {
    __shared__ float scratch[threads_per_row];
    const std::size_t offset = blockIdx.x * columns;
    const float* row = input + offset;
    const float maximum = reduce_row<true>(local_max(row, columns), scratch);
    float total = 0;
    for (std::size_t column = threadIdx.x; column < columns; column += threads_per_row) {
        total += expf(row[column] - maximum);
    }
    const float sum = reduce_row<false>(total, scratch);
    for (std::size_t column = threadIdx.x; column < columns; column += threads_per_row) {
        output[offset + column] = expf(row[column] - maximum) / sum;
    }
}

bool validate_rows(
    const std::vector<float>& input, const std::vector<float>& output, int rows, std::size_t columns
) {
    const int checked_rows = std::min(rows, 32);
    for (int sample = 0; sample < checked_rows; ++sample) {
        const std::size_t row = checked_rows == 1 ? 0
            : static_cast<std::size_t>(sample) * (rows - 1) / (checked_rows - 1);
        const float* values = input.data() + row * columns;
        const double maximum = *std::max_element(values, values + columns);
        double sum = 0;
        for (std::size_t column = 0; column < columns; ++column) {
            sum += std::exp(values[column] - maximum);
        }
        for (std::size_t column = 0; column < columns; ++column) {
            const double reference = std::exp(values[column] - maximum) / sum;
            const float actual = output[row * columns + column];
            if (!std::isfinite(actual) || std::abs(actual - reference) > 2e-6 + 1e-4 * reference) {
                return false;
            }
        }
    }
    return true;
}

}

Measurement softmax(const Options& options, const cudaDeviceProp& properties) {
    const int rows = options.shape[0];
    const std::size_t columns = options.shape[1];
    if (rows > properties.maxGridSize[0]) {
        throw std::invalid_argument("softmax shape exceeds CUDA grid dimensions");
    }
    const std::size_t count = rows * columns;
    const std::size_t bytes = count * sizeof(float);
    DeviceArray device_input(count), device_output(count);
    auto input = input_values(count, options.seed);
    for (std::size_t i = 0; i < count; ++i) {
        input[i] += (i / columns) % 2 == 0 ? 100.0f : -100.0f;
    }
    std::vector<float> output(count);
    cuda_check(cudaMemcpy(device_input.data(), input.data(), bytes, cudaMemcpyHostToDevice), "upload softmax input");

    double latency = 0;
    if (options.variant == "fused") {
        latency = measure(options, [&] {
            softmax_fused<<<rows, threads_per_row>>>(device_input.data(), device_output.data(), columns);
        });
    } else {
        DeviceArray maxima(rows), sums(rows), exponentials(count);
        latency = measure(options, [&] {
            softmax_reduce_max<<<rows, threads_per_row>>>(device_input.data(), maxima.data(), columns);
            softmax_exp_sum<<<rows, threads_per_row>>>(
                device_input.data(), maxima.data(), exponentials.data(), sums.data(), columns
            );
            softmax_normalize<<<rows, threads_per_row>>>(
                exponentials.data(), sums.data(), device_output.data(), columns
            );
        });
    }
    cuda_check(cudaMemcpy(output.data(), device_output.data(), bytes, cudaMemcpyDeviceToHost), "download softmax output");
    return {latency, 2.0 * bytes, validate_rows(input, output, rows, columns)};
}

}
