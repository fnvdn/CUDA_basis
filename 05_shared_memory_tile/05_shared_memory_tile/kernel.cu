#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>

#define N 8
#define BLOCK_SIZE 4

// ==========================================
// GPU kernel：使用 shared memory 做 block 内反转
// ==========================================
//
// input:  输入数组，存放在 GPU global memory 中。
// output: 输出数组，存放在 GPU global memory 中。
// n:      数组长度。
//
// 每个 block 处理 BLOCK_SIZE 个元素。
// 先把数据从 global memory 读到 shared memory，
// 然后在 shared memory 中做局部反转，
// 最后写回 global memory。
//
__global__ void reverse_each_block(const int* input, int* output, int n)
{
    // 声明 shared memory。
    //
    // __shared__ 表示这个数组位于 shared memory 中。
    // tile 被同一个 block 内的所有线程共享。
    //
    // 注意：
    // 每个 block 都有自己独立的一份 tile。
    __shared__ int tile[BLOCK_SIZE];

    // 当前线程在 block 内的编号。
    int local_id = threadIdx.x;

    // 当前线程在整个 grid 中的全局编号。
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    // 从 global memory 读取数据，写入 shared memory。
    //
    // input[global_id] 在 global memory 中。
    // tile[local_id] 在 shared memory 中。
    if (global_id < n) {
        tile[local_id] = input[global_id];
    }

    // 同步同一个 block 内的所有线程。
    //
    // 必须等所有线程都把数据写入 tile 后，
    // 才能让线程继续从 tile 中读取数据。
    //
    // 如果没有这一步，某些线程可能会提前读取，
    // 读到还没有写好的 shared memory 数据。
    __syncthreads();

    // 在 block 内做反转。
    //
    // local_id = 0 读取 tile[3]
    // local_id = 1 读取 tile[2]
    // local_id = 2 读取 tile[1]
    // local_id = 3 读取 tile[0]
    //
    // 这样 block 内的 4 个元素就被反转了。
    int reversed_local_id = blockDim.x - 1 - local_id;

    if (global_id < n) {
        output[global_id] = tile[reversed_local_id];
    }
}

void print_array(const char* name, const int* data, int n)
{
    printf("%s: ", name);
    for (int i = 0; i < n; i++) {
        printf("%d ", data[i]);
    }
    printf("\n");
}

int main()
{
    const int size = N * sizeof(int);

    // CPU 端输入数组。
    int h_input[N] = { 1, 2, 3, 4, 5, 6, 7, 8 };

    // CPU 端输出数组。
    int h_output[N] = { 0 };

    // GPU 端数组指针。
    int* d_input = nullptr;
    int* d_output = nullptr;

    // 在 GPU global memory 中分配空间。
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    // 把输入数组从 CPU 复制到 GPU。
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // 设置线程组织。
    //
    // 每个 block 有 4 个线程。
    // N = 8，所以需要 2 个 block。
    int threads_per_block = BLOCK_SIZE;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    // 启动 kernel。
    reverse_each_block << <blocks_per_grid, threads_per_block >> > (d_input, d_output, N);

    // 等待 GPU 执行完成，并检查错误。
    cudaError_t sync_status = cudaDeviceSynchronize();
    if (sync_status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(sync_status));
        return 1;
    }

    // 把结果从 GPU 复制回 CPU。
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // 打印输入和输出。
    print_array("Input ", h_input, N);
    print_array("Output", h_output, N);
    printf("Expected output: 4 3 2 1 8 7 6 5\n");

    // 释放 GPU 显存。
    cudaFree(d_input);
    cudaFree(d_output);

    cudaDeviceReset();

    return 0;
}