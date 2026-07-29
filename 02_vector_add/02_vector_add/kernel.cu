#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>

// GPU kernel 函数：在 GPU 上执行。
// 作用：让多个 GPU 线程并行完成向量加法。
// a: 输入向量 A，存放在 GPU 显存中。
// b: 输入向量 B，存放在 GPU 显存中。
// c: 输出向量 C，存放在 GPU 显存中。
// n: 向量长度。
__global__ void vector_add(const int* a, const int* b, int* c, int n)
{
    // 计算当前线程在整个 grid 中的全局编号。
    //
    // blockIdx.x：当前 block 的编号。
    // blockDim.x：每个 block 中有多少个线程。
    // threadIdx.x：当前线程在 block 内的编号。
    //
    // 公式：
    // 全局线程编号 = block 编号 * 每个 block 的线程数 + block 内线程编号
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 每个线程负责计算一个数组元素：
    // 线程 i 计算 c[i] = a[i] + b[i]。
    //
    // if (i < n) 是边界保护。
    // 因为启动的线程数可能大于数组长度 n，
    // 多出来的线程不能访问数组越界位置。
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    // 向量长度。
    // 这里为了方便观察，只设置 8 个元素。
    const int n = 8;

    // 需要分配的内存字节数。
    // int 是 4 字节，所以总大小是 n * sizeof(int)。
    const int size = n * sizeof(int);

    // h_ 前缀表示 host，也就是 CPU 端内存中的数据。
    // h_a 和 h_b 是输入向量。
    // h_c 用来接收 GPU 计算后的结果。
    int h_a[n] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    int h_b[n] = { 10, 20, 30, 40, 50, 60, 70, 80 };
    int h_c[n] = { 0 };

    // d_ 前缀表示 device，也就是 GPU 端显存中的数据。
    // GPU kernel 不能直接使用普通 CPU 数组 h_a、h_b、h_c。
    // 所以需要在 GPU 显存中再分配 d_a、d_b、d_c。
    int* d_a = nullptr;
    int* d_b = nullptr;
    int* d_c = nullptr;

    // 在 GPU 显存中分配空间。
    // d_a 用来存放向量 A。
    // d_b 用来存放向量 B。
    // d_c 用来存放计算结果 C。
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // 把 CPU 内存中的 h_a 拷贝到 GPU 显存 d_a。
    // cudaMemcpyHostToDevice 表示从 Host 到 Device。
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    // 把 CPU 内存中的 h_b 拷贝到 GPU 显存 d_b。
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // 每个 block 中放 4 个线程。
    int threads_per_block = 4;

    // 计算需要多少个 block。
    //
    // 公式：
    // blocks_per_grid = 向上取整(n / threads_per_block)
    //
    // 这里 n = 8，threads_per_block = 4，
    // 所以 blocks_per_grid = 2。
    //
    // 如果 n = 10，threads_per_block = 4，
    // 那么 blocks_per_grid = 3，
    // 因为 2 个 block 只有 8 个线程，不够处理 10 个元素。
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    // 启动 GPU kernel。
    //
    // <<<blocks_per_grid, threads_per_block>>> 表示：
    // 启动 blocks_per_grid 个 block，
    // 每个 block 中有 threads_per_block 个线程。
    //
    // 本例中是：
    // <<<2, 4>>>
    //
    // 总共 2 * 4 = 8 个线程。
    // 每个线程负责计算一个元素。
    vector_add << <blocks_per_grid, threads_per_block >> > (d_a, d_b, d_c, n);

    // 等待 GPU kernel 执行完成。
    // CUDA kernel 启动通常是异步的，
    // CPU 发出启动命令后会继续往下走。
    // 所以这里需要同步，确保 GPU 已经算完。
    cudaError_t sync_status = cudaDeviceSynchronize();

    // 检查 kernel 执行是否出错。
    if (sync_status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(sync_status));
        return 1;
    }

    // 把 GPU 显存中的计算结果 d_c 拷贝回 CPU 内存 h_c。
    // cudaMemcpyDeviceToHost 表示从 Device 到 Host。
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // 在 CPU 端打印计算结果。
    printf("A + B = ");
    for (int i = 0; i < n; i++) {
        printf("%d ", h_c[i]);
    }
    printf("\n");

    // 释放 GPU 显存。
    // cudaMalloc 分配的显存必须用 cudaFree 释放。
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // 重置 GPU 设备，清理 CUDA 运行时状态。
    cudaDeviceReset();

    return 0;
}