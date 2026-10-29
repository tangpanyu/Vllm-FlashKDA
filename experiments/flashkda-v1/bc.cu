#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace {

constexpr int kWarpSize = 32;
constexpr int kBlocks = 1024;
constexpr int kIterations = 256;
constexpr int kConflictStride = 32;
constexpr int kSharedInts = (kWarpSize - 1) * kConflictStride + 1;

void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// One 32-bit word per lane:
//   bank(lane) = lane % 32
// Every lane accesses a different bank.
__global__ void no_bank_conflict(int* output, int iterations) {
    __shared__ int storage[kWarpSize];
    volatile int* smem = storage;

    const int lane = threadIdx.x & (kWarpSize - 1);
    int sum = 0;

    for (int i = 0; i < iterations; ++i) {
        smem[lane] = lane + i;
        __syncwarp();
        sum += smem[lane];
        __syncwarp();
    }

    output[blockIdx.x * kWarpSize + lane] = sum;
}

// 32 words between neighboring lanes:
//   bank(lane) = (lane * 32) % 32 = 0
// All lanes access different words in bank 0, producing a 32-way conflict.
__global__ void bank_conflict_32way(int* output, int iterations) {
    __shared__ int storage[kSharedInts];
    volatile int* smem = storage;

    const int lane = threadIdx.x & (kWarpSize - 1);
    const int index = lane * kConflictStride;
    int sum = 0;

    for (int i = 0; i < iterations; ++i) {
        smem[index] = lane + i;
        __syncwarp();
        sum += smem[index];
        __syncwarp();
    }

    output[blockIdx.x * kWarpSize + lane] = sum;
}

}  // namespace

int main() {
    constexpr int kOutputElements = kBlocks * kWarpSize;
    constexpr size_t kOutputBytes = kOutputElements * sizeof(int);

    int* device_output = nullptr;
    check_cuda(cudaMalloc(&device_output, kOutputBytes), "cudaMalloc");

    no_bank_conflict<<<kBlocks, kWarpSize>>>(device_output, kIterations);
    check_cuda(cudaGetLastError(), "no_bank_conflict launch");
    check_cuda(cudaDeviceSynchronize(), "no_bank_conflict execution");

    bank_conflict_32way<<<kBlocks, kWarpSize>>>(device_output, kIterations);
    check_cuda(cudaGetLastError(), "bank_conflict_32way launch");
    check_cuda(cudaDeviceSynchronize(), "bank_conflict_32way execution");

    int first_result = 0;
    check_cuda(
        cudaMemcpy(&first_result, device_output, sizeof(first_result), cudaMemcpyDeviceToHost),
        "cudaMemcpy");
    check_cuda(cudaFree(device_output), "cudaFree");

    const int expected = kIterations * (kIterations - 1) / 2;
    std::printf("Finished no_bank_conflict and bank_conflict_32way.\n");
    std::printf("lane 0 result: %d (expected %d)\n", first_result, expected);
    std::printf("\nCompile:\n");
    std::printf("  nvcc -O3 -lineinfo -arch=sm_120 bc.cu -o bc\n");
    std::printf("\nProfile separately and compare Memory Workload Analysis:\n");
    std::printf("  ncu --set full --kernel-name-base function \\\n");
    std::printf("      -k 'regex:^no_bank_conflict$' -c 1 -o bc_no_conflict ./bc\n");
    std::printf("  ncu --set full --kernel-name-base function \\\n");
    std::printf("      -k 'regex:^bank_conflict_32way$' -c 1 -o bc_conflict ./bc\n");

    return first_result == expected ? EXIT_SUCCESS : EXIT_FAILURE;
}
