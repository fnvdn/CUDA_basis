#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

/*
    实验名称：
    08_vector_add_timing

    实验目标：
    1. 复习向量加法 C[i] = A[i] + B[i]。
    2. 使用 CUDA event 测量 kernel 执行时间。
    3. 对比不同数据规模下 GPU kernel 的耗时。
    4. 初步理解 GPU 吞吐量和内存带宽。

    这个实验和前面的区别：

    之前我们更关心“结果对不对”。
    现在我们开始关心“运行要多久”。
*/

#define BLOCK_SIZE 256

/*
    CUDA kernel：向量加法

    每个线程负责计算一个数组元素：

        C[i] = A[i] + B[i]

    A、B、C 都在 GPU global memory 中。
*/
__global__ void vector_add_kernel(const float* A, const float* B, float* C, int n)
{
    /*
        计算当前线程的全局编号。

        blockIdx.x：
            当前 block 在 grid 中的编号。

        blockDim.x：
            每个 block 中的线程数量。

        threadIdx.x：
            当前线程在 block 内的编号。
    */
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    /*
        防止线程编号超过数组长度。

        因为 blocks_per_grid 是向上取整得到的，
        可能会多启动几个线程。
    */
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

/*
    初始化 CPU 数组。

    data:
        要初始化的数组。

    n:
        数组长度。

    value:
        每个元素要设置成的值。
*/
void init_array(float* data, int n, float value)
{
    for (int i = 0; i < n; i++) {
        data[i] = value;
    }
}

/*
    CUDA 错误检查函数。

    如果某个 CUDA API 调用失败，
    就打印错误信息。
*/
void check_cuda(cudaError_t status, const char* message)
{
    if (status != cudaSuccess) {
        printf("CUDA error at %s: %s\n", message, cudaGetErrorString(status));
        exit(1);
    }
}

/*
    run_test 函数：

    对给定的数组长度 n，完成一次完整测试：

        1. CPU 分配数组
        2. GPU 分配数组
        3. CPU -> GPU 复制 A、B
        4. 用 CUDA event 测量 kernel 时间
        5. GPU -> CPU 复制 C
        6. 打印耗时和一个样本结果
        7. 释放内存
*/
void run_test(int n)
{
    /*
        计算数组占用的字节数。

        float 是 4 字节。
        如果 n = 1,000,000，
        一个数组大约占 4 MB。
    */
    int bytes = n * sizeof(float);

    /*
        在 CPU 内存中分配 A、B、C。

        h_ 表示 host，也就是 CPU。
    */
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);

    /*
        初始化 CPU 数组。

        A 全部为 1.0
        B 全部为 2.0

        所以理论上：
        C[i] = 3.0
    */
    init_array(h_A, n, 1.0f);
    init_array(h_B, n, 2.0f);

    /*
        GPU 端数组指针。

        d_ 表示 device，也就是 GPU。
    */
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    /*
        在 GPU global memory 中分配空间。
    */
    check_cuda(cudaMalloc((void**)&d_A, bytes), "cudaMalloc d_A");
    check_cuda(cudaMalloc((void**)&d_B, bytes), "cudaMalloc d_B");
    check_cuda(cudaMalloc((void**)&d_C, bytes), "cudaMalloc d_C");

    /*
        把 A、B 从 CPU 复制到 GPU。

        注意：
        本实验测量的是 kernel 运行时间，
        不把 cudaMemcpy 的时间算进去。
    */
    check_cuda(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice),
        "copy A host to device");
    check_cuda(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice),
        "copy B host to device");

    /*
        配置 kernel 启动参数。

        每个 block 有 256 个线程。
        block 数量向上取整，确保覆盖 n 个元素。
    */
    int threads_per_block = BLOCK_SIZE;
    int blocks_per_grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    /*
        创建 CUDA event。

        start:
            记录 kernel 开始前的时间点。

        stop:
            记录 kernel 结束后的时间点。
    */
    cudaEvent_t start;
    cudaEvent_t stop;

    check_cuda(cudaEventCreate(&start), "create start event");
    check_cuda(cudaEventCreate(&stop), "create stop event");

    /*
        在 GPU 时间线上记录起点。
    */
    check_cuda(cudaEventRecord(start), "record start event");

    /*
        启动向量加法 kernel。

        注意：
        kernel 启动本身是异步的。
        CPU 发出启动命令后不会自动等待 GPU 完成。
    */
    vector_add_kernel << <blocks_per_grid, threads_per_block >> > (d_A, d_B, d_C, n);

    /*
        检查 kernel 启动有没有出错。

        例如：
        grid/block 参数错误、非法配置等。
    */
    check_cuda(cudaGetLastError(), "kernel launch");

    /*
        在 GPU 时间线上记录终点。

        由于 start 和 stop 都在默认流中，
        stop 会等前面的 kernel 执行完后才被记录。
    */
    check_cuda(cudaEventRecord(stop), "record stop event");

    /*
        等待 stop event 完成。

        这表示 kernel 已经执行结束。
    */
    check_cuda(cudaEventSynchronize(stop), "sync stop event");

    /*
        计算 start 到 stop 之间的时间。

        单位是毫秒 ms。
    */
    float elapsed_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
        "calculate elapsed time");

    /*
        把结果 C 从 GPU 复制回 CPU。
    */
    check_cuda(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost),
        "copy C device to host");

    /*
        向量加法的内存访问量：

        每个元素：
            读取 A[i]：4 字节
            读取 B[i]：4 字节
            写入 C[i]：4 字节

        总共约 12 字节。

        所以总访问量约为：
            n * 12 字节

        带宽 GB/s：
            总字节数 / 时间

        注意：
        这是一个粗略估计，用来帮助理解内存带宽。
    */
    double total_bytes = (double)n * 3.0 * sizeof(float);
    double seconds = elapsed_ms / 1000.0;
    double bandwidth_gb_s = total_bytes / seconds / 1e9;

    /*
        打印结果。

        h_C[0] 应该接近 3.0。
    */
    printf("n = %d\n", n);
    printf("kernel time = %.6f ms\n", elapsed_ms);
    printf("estimated bandwidth = %.2f GB/s\n", bandwidth_gb_s);
    printf("sample h_C[0] = %.1f\n", h_C[0]);
    printf("----------------------------------------\n");

    /*
        释放 event。
    */
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    /*
        释放 GPU 内存。
    */
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    /*
        释放 CPU 内存。
    */
    free(h_A);
    free(h_B);
    free(h_C);
}

int main()
{
    /*
        打印实验说明。
    */
    printf("08_vector_add_timing\n");
    printf("Measure CUDA vector addition kernel time.\n");
    printf("----------------------------------------\n");

    /*
        测试不同数据规模。

        1 << 10 = 1024
        1 << 16 = 65536
        1 << 20 = 1048576
        1 << 24 = 16777216

        数据越大，kernel 时间通常越明显。
    */
    run_test(1 << 10);
    run_test(1 << 16);
    run_test(1 << 20);
    run_test(1 << 24);

    /*
        重置 GPU 设备。

        方便 Visual Studio 调试和 CUDA profiler 正确收尾。
    */
    cudaDeviceReset();

    return 0;
}