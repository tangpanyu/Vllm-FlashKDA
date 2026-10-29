#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kWarpSize = 32;
constexpr int kIterations = 100000;
constexpr int kWarmupIterations = 256;
constexpr int kRepeats = 20;

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t error = (expr);                                              \
        if (error != cudaSuccess) {                                              \
            std::fprintf(stderr, "%s:%d: CUDA error: %s\n",                    \
                         __FILE__, __LINE__, cudaGetErrorString(error));          \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                       \
    } while (false)

template <bool Transpose>
__device__ __forceinline__ void issue_ldmatrix_x4(
    std::uint32_t shared_address,
    std::uint32_t& r0,
    std::uint32_t& r1,
    std::uint32_t& r2,
    std::uint32_t& r3) {
    if constexpr (Transpose) {
        asm volatile(
            "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 "
            "{%0, %1, %2, %3}, [%4];\n"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
            : "r"(shared_address));
    } else {
        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
            "{%0, %1, %2, %3}, [%4];\n"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
            : "r"(shared_address));
    }
}

template <bool Transpose>
__global__ void benchmark_ldmatrix(std::uint64_t* elapsed_cycles,
                                   std::uint32_t* sink) {
    __shared__ alignas(16) std::uint16_t matrix[16][16];

    const int lane = int(threadIdx.x) & 31;
    for (int i = lane; i < 256; i += kWarpSize) {
        matrix[i / 16][i % 16] = static_cast<std::uint16_t>(i);
    }
    __syncwarp();

    // Four 8x8 matrices in column-major tile order:
    //
    //   matrix 0 | matrix 2
    //   ---------+---------
    //   matrix 1 | matrix 3
    const int matrix_id = lane / 8;
    const int row_in_matrix = lane % 8;
    const int tile_row = (matrix_id % 2) * 8;
    const int tile_col = (matrix_id / 2) * 8;
    const void* row_address = &matrix[tile_row + row_in_matrix][tile_col];
    const std::uint32_t shared_address =
        static_cast<std::uint32_t>(__cvta_generic_to_shared(row_address));

    std::uint32_t r0 = 0;
    std::uint32_t r1 = 0;
    std::uint32_t r2 = 0;
    std::uint32_t r3 = 0;

#pragma unroll 1
    for (int i = 0; i < kWarmupIterations; ++i) {
        issue_ldmatrix_x4<Transpose>(shared_address, r0, r1, r2, r3);
    }

    // Keep all lanes aligned at the timing boundary. clock64() is read by the
    // whole warp; lane 0 stores the common elapsed value.
    __syncwarp();
    const std::uint64_t begin = clock64();

#pragma unroll 1
    for (int i = 0; i < kIterations; ++i) {
        issue_ldmatrix_x4<Transpose>(shared_address, r0, r1, r2, r3);
    }

    // Consume the final destination registers before taking the ending clock,
    // so the last ldmatrix cannot remain outstanding past the timed region.
    const std::uint32_t checksum = r0 ^ r1 ^ r2 ^ r3;
    asm volatile("" : "+r"(r0), "+r"(r1), "+r"(r2), "+r"(r3));
    const std::uint64_t end = clock64();

    if (lane == 0) {
        *elapsed_cycles = end - begin;
        *sink = checksum;
    }
}

template <bool Transpose>
std::vector<std::uint64_t> run_benchmark(std::uint64_t* device_cycles,
                                         std::uint32_t* device_sink) {
    std::vector<std::uint64_t> samples;
    samples.reserve(kRepeats);

    for (int repeat = 0; repeat < kRepeats; ++repeat) {
        benchmark_ldmatrix<Transpose><<<1, kWarpSize>>>(device_cycles,
                                                        device_sink);
        CUDA_CHECK(cudaGetLastError());

        std::uint64_t cycles = 0;
        CUDA_CHECK(cudaMemcpy(&cycles, device_cycles, sizeof(cycles),
                              cudaMemcpyDeviceToHost));
        samples.push_back(cycles);
    }
    return samples;
}

void print_result(const char* name, std::vector<std::uint64_t> samples) {
    std::sort(samples.begin(), samples.end());
    const std::uint64_t minimum = samples.front();
    const std::uint64_t median =
        (samples[kRepeats / 2 - 1] + samples[kRepeats / 2]) / 2;

    std::printf("%-20s min %10llu cycles  %8.4f cycles/ldmatrix.x4\n",
                name,
                static_cast<unsigned long long>(minimum),
                double(minimum) / kIterations);
    std::printf("%-20s med %10llu cycles  %8.4f cycles/ldmatrix.x4\n",
                "",
                static_cast<unsigned long long>(median),
                double(median) / kIterations);

    std::printf("  samples:");
    for (std::uint64_t cycles : samples) {
        std::printf(" %.4f", double(cycles) / kIterations);
    }
    std::printf("\n");
}

}  // namespace

int main() {
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));

    std::uint64_t* device_cycles = nullptr;
    std::uint32_t* device_sink = nullptr;
    CUDA_CHECK(cudaMalloc(&device_cycles, sizeof(*device_cycles)));
    CUDA_CHECK(cudaMalloc(&device_sink, sizeof(*device_sink)));

    // An untimed launch initializes the CUDA context before collecting data.
    benchmark_ldmatrix<false><<<1, kWarpSize>>>(device_cycles, device_sink);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto normal = run_benchmark<false>(device_cycles, device_sink);
    const auto transposed = run_benchmark<true>(device_cycles, device_sink);

    std::printf("GPU: %s, SM %d.%d, clockRate reported %d kHz\n",
                properties.name,
                properties.major,
                properties.minor,
                properties.clockRate);
    std::printf("one warp, %d warmups, %d timed iterations, %d repeats\n\n",
                kWarmupIterations,
                kIterations,
                kRepeats);
    print_result("ldmatrix.x4", normal);
    print_result("ldmatrix.x4.trans", transposed);

    CUDA_CHECK(cudaFree(device_cycles));
    CUDA_CHECK(cudaFree(device_sink));
}
