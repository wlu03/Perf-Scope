#include "benchmark.hpp"

namespace perf_scope {
namespace {

constexpr int tile_width = 32;
constexpr int block_rows = 8;

__global__ void transpose_naive(const float* input, float* output, int rows, int columns) {
    const int column = blockIdx.x * tile_width + threadIdx.x;
    const int first_row = blockIdx.y * tile_width + threadIdx.y;
    for (int offset = 0; offset < tile_width; offset += block_rows) {
        const int row = first_row + offset;
        if (row < rows && column < columns) {
            output[static_cast<std::size_t>(column) * rows + row] =
                input[static_cast<std::size_t>(row) * columns + column];
        }
    }
}

__global__ void transpose_tiled(const float* input, float* output, int rows, int columns) {
    __shared__ float tile[tile_width][tile_width + 1];
    const int column = blockIdx.x * tile_width + threadIdx.x;
    const int first_row = blockIdx.y * tile_width + threadIdx.y;
    for (int offset = 0; offset < tile_width; offset += block_rows) {
        const int row = first_row + offset;
        if (row < rows && column < columns) {
            tile[threadIdx.y + offset][threadIdx.x] =
                input[static_cast<std::size_t>(row) * columns + column];
        }
    }
    __syncthreads();
    const int output_column = blockIdx.y * tile_width + threadIdx.x;
    const int first_output_row = blockIdx.x * tile_width + threadIdx.y;
    for (int offset = 0; offset < tile_width; offset += block_rows) {
        const int output_row = first_output_row + offset;
        if (output_row < columns && output_column < rows) {
            output[static_cast<std::size_t>(output_row) * rows + output_column] =
                tile[threadIdx.x][threadIdx.y + offset];
        }
    }
}

}

Measurement transpose(const Options& options, const cudaDeviceProp& properties) {
    const int rows = options.shape[0];
    const int columns = options.shape[1];
    const dim3 blocks((columns - 1) / tile_width + 1, (rows - 1) / tile_width + 1);
    const dim3 threads(tile_width, block_rows);
    if (blocks.x > static_cast<unsigned>(properties.maxGridSize[0]) ||
        blocks.y > static_cast<unsigned>(properties.maxGridSize[1])) {
        throw std::invalid_argument("transpose shape exceeds CUDA grid dimensions");
    }
    const std::size_t count = static_cast<std::size_t>(rows) * columns;
    const std::size_t bytes = count * sizeof(float);
    DeviceArray device_input(count), device_output(count);
    const auto input = input_values(count, options.seed);
    std::vector<float> output(count);
    cuda_check(cudaMemcpy(device_input.data(), input.data(), bytes, cudaMemcpyHostToDevice), "upload input");
    const double latency = options.variant == "naive"
        ? measure(options, [&] {
            transpose_naive<<<blocks, threads>>>(device_input.data(), device_output.data(), rows, columns);
        })
        : measure(options, [&] {
            transpose_tiled<<<blocks, threads>>>(device_input.data(), device_output.data(), rows, columns);
        });
    cuda_check(cudaMemcpy(output.data(), device_output.data(), bytes, cudaMemcpyDeviceToHost), "download output");

    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            if (output[static_cast<std::size_t>(column) * rows + row] !=
                input[static_cast<std::size_t>(row) * columns + column]) {
                return {latency, 2.0 * bytes, false};
            }
        }
    }
    return {latency, 2.0 * bytes, true};
}

}
