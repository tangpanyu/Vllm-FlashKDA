#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr int kRows = 16;
constexpr int kCols = 16;
constexpr int kWarpSize = 32;
constexpr int kRegistersPerLane = 4;

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t error = (expr);                                              \
        if (error != cudaSuccess) {                                              \
            std::fprintf(stderr, "%s:%d: CUDA error: %s\n",                    \
                         __FILE__, __LINE__, cudaGetErrorString(error));          \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                       \
    } while (false)

__global__ void load_nmajor_with_trans(std::uint32_t* output_registers) {
    // Logical shape is (M, N). N has unit stride:
    //   offset(m, n) = m * 16 + n
    __shared__ alignas(16) std::uint16_t matrix[kRows * kCols];

    const int lane = int(threadIdx.x) & 31;
    for (int linear = lane; linear < kRows * kCols; linear += kWarpSize) {
        const int m = linear / kCols;
        const int n = linear % kCols;
        // Start from A(m,n) = m*16+n, but write A^T into shared memory:
        // shared(m,n) = A(n,m) = n*16+m.
        matrix[m * kCols + n] = static_cast<std::uint16_t>(n * kRows + m);
    }
    __syncwarp();

    // Four 8x8 matrices in column-major tile order:
    //
    //   matrix 0 | matrix 2
    //   ---------+---------
    //   matrix 1 | matrix 3
    //
    // Each group of eight lanes supplies the row addresses of one 8x8 tile.
    const int matrix_id = lane / 8;
    const int row_in_matrix = lane % 8;
    const int tile_m = (matrix_id % 2) * 8;
    const int tile_n = (matrix_id / 2) * 8;
    const void* row_address =
        &matrix[(tile_m + row_in_matrix) * kCols + tile_n];
    const std::uint32_t shared_address =
        static_cast<std::uint32_t>(__cvta_generic_to_shared(row_address));

    std::uint32_t r0, r1, r2, r3;
    asm volatile(
        "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 "
        "{%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(shared_address));

    // Direct register writeback: no element rearrangement on the device.
    output_registers[lane * kRegistersPerLane + 0] = r0;
    output_registers[lane * kRegistersPerLane + 1] = r1;
    output_registers[lane * kRegistersPerLane + 2] = r2;
    output_registers[lane * kRegistersPerLane + 3] = r3;
}

void print_thread_zero(const std::uint32_t* registers) {
    const auto* b16 = reinterpret_cast<const std::uint16_t*>(registers);

    std::printf("Original A: row-major uint16_t[16][16], values 0..255\n");
    std::printf("Shared input: A^T, first row = 0,16,32,...,240\n");
    std::printf("thread 0 direct registers after ldmatrix.x4.trans:\n");
    std::printf("r0=(%u, %u)  r1=(%u, %u)  r2=(%u, %u)  r3=(%u, %u)\n",
                unsigned(b16[0]), unsigned(b16[1]),
                unsigned(b16[2]), unsigned(b16[3]),
                unsigned(b16[4]), unsigned(b16[5]),
                unsigned(b16[6]), unsigned(b16[7]));
}

}  // namespace

int main() {
    constexpr std::size_t kBytes =
        kWarpSize * kRegistersPerLane * sizeof(std::uint32_t);

    std::uint32_t host_registers[kWarpSize * kRegistersPerLane] = {};
    std::uint32_t* device_registers = nullptr;
    CUDA_CHECK(cudaMalloc(&device_registers, kBytes));

    load_nmajor_with_trans<<<1, kWarpSize>>>(device_registers);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_registers, device_registers, kBytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device_registers));

    print_thread_zero(host_registers);
}
