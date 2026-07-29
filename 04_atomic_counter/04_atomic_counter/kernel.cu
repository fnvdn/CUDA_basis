#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>

#define N 16

// ==========================================
// GPU kernel：使用 atomicAdd 统计正数个数
// ==========================================
//
// data: 输入数组，存放在 GPU 显存中。
// count: 计数器，存放在 GPU 显存中。
// n: 数组长度。
//
// 目标：统计 data 中有多少个元素大于 0。
//
__global__ void count_positive_atomic(const int* data, int* count, int n)
{
    // 计算当前线程的全局编号。
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查，防止线程访问超过数组范围。
    if (i < n) {
        // 如果当前线程负责的元素大于 0，
        // 就把全局计数器 count 加 1。
        //
        // 这里必须用 atomicAdd。
        // 因为很多线程可能同时执行：
        // *count = *count + 1
        //
        // 普通加法会发生竞争冲突。
        // atomicAdd 可以保证每次加 1 都是完整且不可被打断的。
        if (data[i] > 0) {
            atomicAdd(count, 1);
        }
    }
}

int main()
{
    const int size = N * sizeof(int);

    // h_ 表示 host，也就是 CPU 内存。
    //
    // 这个数组里正数有：
    // 3, 5, 7, 2, 8, 1, 6, 4
    //
    // 一共 8 个正数。
    int h_data[N] = {
        -1,  3, -2,  5,
         0,  7, -4,  2,
         8, -9,  1, -6,
         6,  4, -3,  0
    };

    int h_count = 0;

    // d_ 表示 device，也就是 GPU 显存。
    int* d_data = nullptr;
    int* d_count = nullptr;

    // 在 GPU 显存中分配输入数组和计数器。
    cudaMalloc((void**)&d_data, size);
    cudaMalloc((void**)&d_count, sizeof(int));

    // 把输入数组从 CPU 复制到 GPU。
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    // 把计数器初始值 0 从 CPU 复制到 GPU。
    cudaMemcpy(d_count, &h_count, sizeof(int), cudaMemcpyHostToDevice);

    // 设置线程组织。
    //
    // 每个 block 有 4 个线程。
    // N = 16，所以需要 4 个 block。
    int threads_per_block = 4;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    // 启动 kernel。
    // 总共启动 4 * 4 = 16 个线程。
    // 每个线程检查数组中的一个元素。
    count_positive_atomic << <blocks_per_grid, threads_per_block >> > (d_data, d_count, N);

    // 等待 GPU 执行完成。
    cudaError_t sync_status = cudaDeviceSynchronize();
    if (sync_status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(sync_status));
        return 1;
    }

    // 把 GPU 上的计数结果复制回 CPU。
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    // 打印结果。
    printf("Positive count = %d\n", h_count);
    printf("Expected count = 8\n");

    // 释放 GPU 显存。
    cudaFree(d_data);
    cudaFree(d_count);

    cudaDeviceReset();

    return 0;
}