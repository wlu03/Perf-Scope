#include "benchmark.hpp"

#include <cublas_v2.h>

#include <algorithm>
#include <array>

namespace perf_scope {
namespace {

constexpr int tile_width = 16;

__global__ void gemm_naive(
    const float* a, const float* b, float* c, int rows, int columns, int inner
) {
    const int row = blockIdx.y * tile_width + threadIdx.y;
    const int column = blockIdx.x * tile_width + threadIdx.x;
    if (row < rows && column < columns) {
        float sum = 0;
        for (int k = 0; k < inner; ++k) {
            sum = fmaf(a[static_cast<std::size_t>(row) * inner + k],
                       b[static_cast<std::size_t>(k) * columns + column], sum);
        }
        c[static_cast<std::size_t>(row) * columns + column] = sum;
    }
}

__global__ void gemm_tiled(
    const float* a, const float* b, float* c, int rows, int columns, int inner
) {
    __shared__ float tile_a[tile_width][tile_width];
    __shared__ float tile_b[tile_width][tile_width];
    const int row = blockIdx.y * tile_width + threadIdx.y;
    const int column = blockIdx.x * tile_width + threadIdx.x;
    float sum = 0;
    const int tiles = (inner - 1) / tile_width + 1;
    for (int tile = 0; tile < tiles; ++tile) {
        const int a_column = tile * tile_width + threadIdx.x;
        const int b_row = tile * tile_width + threadIdx.y;
        tile_a[threadIdx.y][threadIdx.x] = row < rows && a_column < inner
            ? a[static_cast<std::size_t>(row) * inner + a_column] : 0;
        tile_b[threadIdx.y][threadIdx.x] = b_row < inner && column < columns
            ? b[static_cast<std::size_t>(b_row) * columns + column] : 0;
        __syncthreads();
        for (int k = 0; k < tile_width; ++k) {
            sum = fmaf(tile_a[threadIdx.y][k], tile_b[k][threadIdx.x], sum);
        }
        __syncthreads();
    }
    if (row < rows && column < columns) {
        c[static_cast<std::size_t>(row) * columns + column] = sum;
    }
}

void cublas_check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuBLAS: ") + cublasGetStatusString(status));
    }
}

class BlasHandle {
public:
    BlasHandle() { cublas_check(cublasCreate(&handle_)); }
    ~BlasHandle() { cublasDestroy(handle_); }
    BlasHandle(const BlasHandle&) = delete;
    BlasHandle& operator=(const BlasHandle&) = delete;
    cublasHandle_t native() const { return handle_; }

private:
    cublasHandle_t handle_{};
};

bool validate_product(
    const std::vector<float>& a, const std::vector<float>& b, const std::vector<float>& c,
    int rows, int columns, int inner, std::uint64_t seed
) {
    const std::size_t stride = columns;
    const std::size_t samples = std::min(c.size(), std::size_t{512});
    const std::array<std::size_t, 4> corners{0, stride - 1, (rows - 1) * stride, c.size() - 1};
    std::mt19937_64 rng(seed);
    for (std::size_t sample = 0; sample < samples; ++sample) {
        const std::size_t index = samples == c.size()
            ? sample : (sample < corners.size() ? corners[sample] : rng() % c.size());
        const std::size_t row = index / stride;
        const std::size_t column = index % stride;
        double reference = 0;
        for (int k = 0; k < inner; ++k) {
            reference += static_cast<double>(a[row * inner + k]) *
                         b[static_cast<std::size_t>(k) * stride + column];
        }
        if (!std::isfinite(c[index]) ||
            std::abs(c[index] - reference) > 1e-3 + 1e-4 * std::abs(reference)) {
            return false;
        }
    }
    return true;
}

}

Measurement gemm(const Options& options, const cudaDeviceProp& properties) {
    const int rows = options.shape[0];
    const int columns = options.shape[1];
    const int inner = options.shape[2];
    const dim3 blocks((columns - 1) / tile_width + 1, (rows - 1) / tile_width + 1);
    const dim3 threads(tile_width, tile_width);
    if (blocks.x > static_cast<unsigned>(properties.maxGridSize[0]) ||
        blocks.y > static_cast<unsigned>(properties.maxGridSize[1])) {
        throw std::invalid_argument("GEMM shape exceeds CUDA grid dimensions");
    }
    const std::size_t a_count = static_cast<std::size_t>(rows) * inner;
    const std::size_t b_count = static_cast<std::size_t>(inner) * columns;
    const std::size_t c_count = static_cast<std::size_t>(rows) * columns;
    DeviceArray a_gpu(a_count), b_gpu(b_count), c_gpu(c_count);
    const auto a = input_values(a_count, options.seed);
    const auto b = input_values(b_count, options.seed + 1);
    std::vector<float> c(c_count);
    cuda_check(cudaMemcpy(a_gpu.data(), a.data(), a_count * sizeof(float), cudaMemcpyHostToDevice), "upload A");
    cuda_check(cudaMemcpy(b_gpu.data(), b.data(), b_count * sizeof(float), cudaMemcpyHostToDevice), "upload B");

    double latency = 0;
    int version = 0;
    if (options.variant == "cublas") {
        BlasHandle blas;
        cublas_check(cublasGetVersion(blas.native(), &version));
        const float alpha = 1, beta = 0;
        latency = measure(options, [&] {
            cublas_check(cublasGemmEx(
                blas.native(), CUBLAS_OP_N, CUBLAS_OP_N, columns, rows, inner,
                &alpha, b_gpu.data(), CUDA_R_32F, columns, a_gpu.data(), CUDA_R_32F, inner,
                &beta, c_gpu.data(), CUDA_R_32F, columns, CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT
            ));
        });
    } else if (options.variant == "naive") {
        latency = measure(options, [&] {
            gemm_naive<<<blocks, threads>>>(a_gpu.data(), b_gpu.data(), c_gpu.data(), rows, columns, inner);
        });
    } else {
        latency = measure(options, [&] {
            gemm_tiled<<<blocks, threads>>>(a_gpu.data(), b_gpu.data(), c_gpu.data(), rows, columns, inner);
        });
    }
    cuda_check(cudaMemcpy(c.data(), c_gpu.data(), c_count * sizeof(float), cudaMemcpyDeviceToHost), "download C");
    return {
        latency, 0, validate_product(a, b, c, rows, columns, inner, options.seed),
        2.0 * rows * columns * inner, version,
    };
}

}
