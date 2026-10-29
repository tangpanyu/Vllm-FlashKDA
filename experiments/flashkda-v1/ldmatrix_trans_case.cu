#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr int kWarpSize = 32;
constexpr int kRegistersPerLane = 4;
constexpr int kValuesPerLane = kRegistersPerLane * 2;
constexpr int kMatrixRows = 16;
constexpr int kMatrixCols = 16;

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t error = (expr);                                              \
        if (error != cudaSuccess) {                                              \
            std::fprintf(stderr, "%s:%d: CUDA error: %s\n",                    \
                         __FILE__, __LINE__, cudaGetErrorString(error));          \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                       \
    } while (false)

__device__ __forceinline__ void store_registers_lane_major(
    std::uint32_t* output,
    int lane,
    std::uint32_t r0,
    std::uint32_t r1,
    std::uint32_t r2,
    std::uint32_t r3) {
    // Write the four 32-bit ldmatrix destination registers back verbatim.
    output[lane * kRegistersPerLane + 0] = r0;
    output[lane * kRegistersPerLane + 1] = r1;
    output[lane * kRegistersPerLane + 2] = r2;
    output[lane * kRegistersPerLane + 3] = r3;
}

__global__ void ldmatrix_case(std::uint32_t* normal,
                              std::uint32_t* transposed) {
    __shared__ alignas(16) std::uint16_t matrix[kMatrixRows][kMatrixCols];

    const int lane = static_cast<int>(threadIdx.x) & 31;

    // Fill the row-major 16x16 shared-memory matrix with 0..255.
#pragma unroll
    for (int i = 0; i < kValuesPerLane; ++i) {
        const int linear = lane * kValuesPerLane + i;
        matrix[linear / kMatrixCols][linear % kMatrixCols] =
            static_cast<std::uint16_t>(linear);
    }
    __syncwarp();

    // ldmatrix.x4 sees the 16x16 matrix as four 8x8 matrices. Number the
    // four matrices in column-major tile order, matching CUTE LayoutLeft:
    //
    //   matrix 0 | matrix 2
    //   ---------+---------
    //   matrix 1 | matrix 3
    //
    // Lanes 0..7 provide the eight row addresses for matrix 0, lanes 8..15
    // for matrix 1, lanes 16..23 for matrix 2, and lanes 24..31 for matrix 3.
    const int matrix_id = lane / 8;
    const int row_in_matrix = lane % 8;
    const int tile_row = (matrix_id % 2) * 8;
    const int tile_col = (matrix_id / 2) * 8;
    const void* row_address = &matrix[tile_row + row_in_matrix][tile_col];
    const std::uint32_t shared_address =
        static_cast<std::uint32_t>(__cvta_generic_to_shared(row_address));

    std::uint32_t n0, n1, n2, n3;
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
        "{%0, %1, %2, %3}, [%4];\n"
        : "=r"(n0), "=r"(n1), "=r"(n2), "=r"(n3)
        : "r"(shared_address));

    std::uint32_t t0, t1, t2, t3;
    asm volatile(
        "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 "
        "{%0, %1, %2, %3}, [%4];\n"
        : "=r"(t0), "=r"(t1), "=r"(t2), "=r"(t3)
        : "r"(shared_address));

    // Global memory is deliberately lane-major: thread 0 writes [0,8),
    // thread 1 writes [8,16), ..., and thread 31 writes [248,256).
    store_registers_lane_major(normal, lane, n0, n1, n2, n3);
    store_registers_lane_major(transposed, lane, t0, t1, t2, t3);
}

void print_lane_major(const char* title, const std::uint32_t* registers) {
    const auto* values = reinterpret_cast<const std::uint16_t*>(registers);
    constexpr int kValuesPerLine = 64;
    constexpr int kValueCount = kMatrixRows * kMatrixCols;

    std::printf("\n%s\n", title);
    for (int i = 0; i < kValueCount; ++i) {
        if (i % kValuesPerLine == 0) {
            std::printf("%3d-%3d |", i, i + kValuesPerLine - 1);
        }
        std::printf(" %3u", unsigned(values[i]));
        if (i % kValuesPerLine == kValuesPerLine - 1) {
            std::printf("\n");
        }
    }
}

}  // namespace

int main() {
    constexpr std::size_t kBytes =
        kMatrixRows * kMatrixCols * sizeof(std::uint16_t);

    std::uint32_t normal[kWarpSize * kRegistersPerLane] = {};
    std::uint32_t transposed[kWarpSize * kRegistersPerLane] = {};
    std::uint32_t* device_normal = nullptr;
    std::uint32_t* device_transposed = nullptr;

    CUDA_CHECK(cudaMalloc(&device_normal, kBytes));
    CUDA_CHECK(cudaMalloc(&device_transposed, kBytes));

    ldmatrix_case<<<1, kWarpSize>>>(device_normal, device_transposed);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(normal, device_normal, kBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(transposed, device_transposed, kBytes,
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(device_normal));
    CUDA_CHECK(cudaFree(device_transposed));

    std::printf("shared matrix is row-major uint16_t[16][16], filled with 0..255\n");
    print_lane_major("ldmatrix.x4 (no trans), lane-major writeback:", normal);
    print_lane_major("ldmatrix.x4.trans, lane-major writeback:", transposed);
}
