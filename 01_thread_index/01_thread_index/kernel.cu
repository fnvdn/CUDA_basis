#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>

__global__ void print_thread_index()
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    printf("blockIdx.x=%d, blockDim.x=%d, threadIdx.x=%d, global_id=%d\n",
        blockIdx.x, blockDim.x, threadIdx.x, global_id);
}

int main()
{
    print_thread_index << <3, 4 >> > ();

    cudaError_t sync_status = cudaDeviceSynchronize();
    if (sync_status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(sync_status));
        return 1;
    }

    cudaDeviceReset();

    return 0;
}