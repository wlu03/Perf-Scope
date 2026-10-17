#include "benchmark.hpp"

#include <charconv>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unordered_map>

namespace perf_scope {
namespace {

template <class Integer>
Integer integer(std::string_view input, std::string_view label, Integer minimum) {
    Integer value{};
    const auto [end, error] = std::from_chars(input.data(), input.data() + input.size(), value);
    if (error != std::errc{} || end != input.data() + input.size() || value < minimum) {
        throw std::invalid_argument("invalid " + std::string(label));
    }
    return value;
}

std::vector<int> dimensions(std::string_view text) {
    std::vector<int> result;
    for (;;) {
        const std::size_t comma = text.find(',');
        result.push_back(integer(text.substr(0, comma), "--shape", 1));
        if (comma == std::string_view::npos) {
            return result;
        }
        text.remove_prefix(comma + 1);
    }
}

Options parse(int count, char** argv) {
    if (count != 13) {
        throw std::invalid_argument("expected a workload, variant, and five options");
    }
    std::unordered_map<std::string_view, std::string_view> fields;
    for (int i = 3; i < count; i += 2) {
        if (!fields.emplace(argv[i], argv[i + 1]).second) {
            throw std::invalid_argument("duplicate option " + std::string(argv[i]));
        }
    }
    auto get = [&](std::string_view name) {
        const auto field = fields.find(name);
        if (field == fields.end()) {
            throw std::invalid_argument("missing option " + std::string(name));
        }
        return field->second;
    };
    if (get("--dtype") != "fp32") {
        throw std::invalid_argument("CUDA benchmarks require fp32");
    }
    Options options{
        argv[1], argv[2], dimensions(get("--shape")),
        integer(get("--seed"), "--seed", std::uint64_t{0}),
        integer(get("--warmups"), "--warmups", 0),
        integer(get("--iterations"), "--iterations", 1),
    };
    if (options.workload == "transpose") {
        if (options.shape.size() == 2 && (options.variant == "naive" || options.variant == "tiled")) {
            return options;
        }
        throw std::invalid_argument("transpose requires naive or tiled and shape rows,columns");
    }
    if (options.workload == "gemm") {
        if (options.shape.size() == 3 &&
            (options.variant == "naive" || options.variant == "tiled" || options.variant == "cublas")) {
            return options;
        }
        throw std::invalid_argument("gemm requires naive, tiled, or cublas and shape M,N,K");
    }
    if (options.workload == "triad" && options.variant == "streaming" && options.shape.size() == 1) {
        return options;
    }
    if (options.workload == "softmax" && options.shape.size() == 2 &&
        (options.variant == "fused" || options.variant == "multi-pass")) {
        return options;
    }
    throw std::invalid_argument("expected transpose, gemm, triad streaming, or softmax");
}

std::string device_id(const cudaUUID_t& uuid) {
    std::ostringstream id;
    id << "GPU-" << std::hex << std::setfill('0');
    for (std::size_t i = 0; i < sizeof(uuid.bytes); ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) {
            id << '-';
        }
        id << std::setw(2) << static_cast<unsigned>(static_cast<unsigned char>(uuid.bytes[i]));
    }
    return id.str();
}

Measurement benchmark(const Options& options, const cudaDeviceProp& properties) {
    if (options.workload == "transpose") {
        return transpose(options, properties);
    }
    if (options.workload == "gemm") {
        return gemm(options, properties);
    }
    if (options.workload == "softmax") {
        return softmax(options, properties);
    }
    return triad(options, properties);
}

int run(int count, char** argv) {
    if (count == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << "perf-scope-cuda transpose {naive|tiled} | gemm {naive|tiled|cublas} | "
                     "triad streaming | softmax {fused|multi-pass}\n"
                     "--shape DIMS "
                     "--dtype fp32 --seed SEED --warmups N --iterations N\n";
        return 0;
    }
    const Options options = parse(count, argv);
    int device = 0, driver = 0, runtime = 0;
    cudaDeviceProp properties{};
    cuda_check(cudaGetDevice(&device), "select CUDA device");
    cuda_check(cudaGetDeviceProperties(&properties, device), "read device properties");
    cuda_check(cudaDriverGetVersion(&driver), "read CUDA driver version");
    cuda_check(cudaRuntimeGetVersion(&runtime), "read CUDA runtime version");
    const Measurement result = benchmark(options, properties);
    std::cout << std::setprecision(10) << std::boolalpha
              << "{\"device\":" << std::quoted(device_id(properties.uuid))
              << ",\"correct\":" << result.correct
              << ",\"metrics\":{\"latency_ms\":" << result.latency_ms;
    if (result.logical_bytes > 0) {
        std::cout << ",\"bandwidth_gb_s\":" << result.logical_bytes / (1.0e6 * result.latency_ms);
    }
    if (result.flops > 0) {
        std::cout << ",\"gflops\":" << result.flops / (1.0e6 * result.latency_ms);
    }
    std::cout << "},\"logical_bytes\":" << result.logical_bytes << ",\"flops\":" << result.flops
              << ",\"cuda\":{\"name\":" << std::quoted(properties.name)
              << ",\"major\":" << properties.major << ",\"minor\":" << properties.minor
              << ",\"l2_bytes\":" << properties.l2CacheSize
              << ",\"driver\":" << driver << ",\"runtime\":" << runtime;
    if (result.cublas_version > 0) {
        std::cout << ",\"cublas_version\":" << result.cublas_version
                  << ",\"cublas_compute\":\"CUBLAS_COMPUTE_32F_PEDANTIC\"";
    }
    std::cout
              << "},\"timing_scope\":\"cuda_event\"}\n";
    return 0;
}

}
}

int main(int argc, char** argv) {
    try {
        return perf_scope::run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
