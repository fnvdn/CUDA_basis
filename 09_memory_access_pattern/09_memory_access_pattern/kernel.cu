#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

/*
    实验名称：
    09_memory_access_pattern

    实验目标：
    1. 对比不同 global memory 访问模式的性能。
    2. 理解连续访问为什么更容易获得高带宽。
    3. 初步理解 memory coalescing，也就是访存合并。
    4. 为后面理解 MoE expert/cache/prefetch 的访存模式做铺垫。

    三种访问方式：

    1. 连续访问 contiguous access：
        线程 i 访问 A[i]

    2. 跨步访问 strided access：
        线程 i 访问 A[i * STRIDE]

    3. 反向访问 reverse access：
        线程 i 访问 A[n - 1 - i]
*/

#define N (1 << 24)
#define BLOCK_SIZE 256
#define STRIDE 4

/*
    连续访问 kernel

    每个线程访问相邻位置：

        线程 0 -> A[0]
        线程 1 -> A[1]
        线程 2 -> A[2]
        ...

    这种模式最适合 GPU 做访存合并。
*/
__global__ void contiguous_kernel(const float* A, float* C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        C[i] = A[i];
    }
}

/*
    跨步访问 kernel

    每个线程访问间隔 STRIDE 的位置：

        线程 0 -> A[0]
        线程 1 -> A[4]
        线程 2 -> A[8]
        ...

    线程访问的地址不再连续，
    对 global memory 访问合并不友好。
*/
__global__ void strided_kernel(const float* A, float* C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    /*
        因为访问 A[i * STRIDE]，
        所以 i 必须满足：

            i * STRIDE < n

        即：
            i < n / STRIDE
    */
    if (i < n / STRIDE) {
        C[i] = A[i * STRIDE];
    }
}

/*
    反向访问 kernel

    每个线程访问反向地址：

        线程 0 -> A[n - 1]
        线程 1 -> A[n - 2]
        线程 2 -> A[n - 3]
        ...

    注意：
    虽然方向反了，但相邻线程访问的地址仍然相邻。
    所以它通常仍然比跨步访问友好。
*/
__global__ void reverse_kernel(const float* A, float* C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        C[i] = A[n - 1 - i];
    }
}

/*
    初始化 CPU 数组。
*/
void init_array(float* data, int n)
{
    for (int i = 0; i < n; i++) {
        data[i] = (float)i;
    }
}

/*
    CUDA 错误检查。
*/
void check_cuda(cudaError_t status, const char* message)
{
    if (status != cudaSuccess) {
        printf("CUDA error at %s: %s\n", message, cudaGetErrorString(status));
        exit(1);
    }
}

/*
    测量一个 kernel 的运行时间。

    参数说明：

    name:
        测试名称，例如 contiguous / strided / reverse。

    mode:
        0 表示连续访问。
        1 表示跨步访问。
        2 表示反向访问。

    d_A, d_C:
        GPU 上的输入输出数组。

    n:
        数组长度。
*/
void run_kernel_test(const char* name, int mode, const float* d_A, float* d_C, int n)
{
    int threads_per_block = BLOCK_SIZE;
    int blocks_per_grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    /*
        对于跨步访问，实际输出元素数量是 n / STRIDE。
        但为了保持代码简单，仍然启动足够多线程，
        kernel 内部会用 if 判断越界。
    */

    cudaEvent_t start;
    cudaEvent_t stop;

    check_cuda(cudaEventCreate(&start), "create start event");
    check_cuda(cudaEventCreate(&stop), "create stop event");

    /*
        记录开始时间。
    */
    check_cuda(cudaEventRecord(start), "record start event");

    /*
        根据 mode 选择不同访问模式的 kernel。
    */
    if (mode == 0) {
        contiguous_kernel << <blocks_per_grid, threads_per_block >> > (d_A, d_C, n);
    }
    else if (mode == 1) {
        strided_kernel << <blocks_per_grid, threads_per_block >> > (d_A, d_C, n);
    }
    else if (mode == 2) {
        reverse_kernel << <blocks_per_grid, threads_per_block >> > (d_A, d_C, n);
    }

    check_cuda(cudaGetLastError(), "kernel launch");

    /*
        记录结束时间。
    */
    check_cuda(cudaEventRecord(stop), "record stop event");
    check_cuda(cudaEventSynchronize(stop), "sync stop event");

    float elapsed_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed time");

    /*
        粗略估算访问字节数。

        连续访问：
            读取 A[i]，写入 C[i]
            每个元素 2 次 float 访问 = 8 字节

        反向访问：
            同样读一个 A，写一个 C = 8 字节

        跨步访问：
            实际只处理 n / STRIDE 个输出元素
            所以按 n / STRIDE 估算访问量
    */
    int active_n = n;

    if (mode == 1) {
        active_n = n / STRIDE;
    }

    double total_bytes = (double)active_n * 2.0 * sizeof(float);
    double seconds = elapsed_ms / 1000.0;
    double bandwidth_gb_s = total_bytes / seconds / 1e9;

    printf("%s\n", name);
    printf("kernel time = %.6f ms\n", elapsed_ms);
    printf("estimated bandwidth = %.2f GB/s\n", bandwidth_gb_s);
    printf("----------------------------------------\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main()
{
    printf("09_memory_access_pattern\n");
    printf("Compare different global memory access patterns.\n");
    printf("----------------------------------------\n");

    int bytes = N * sizeof(float);

    /*
        CPU 端输入数组。
    */
    float* h_A = (float*)malloc(bytes);

    /*
        初始化输入数据：
            A[i] = i
    */
    init_array(h_A, N);

    /*
        GPU 端数组。
    */
    float* d_A = nullptr;
    float* d_C = nullptr;

    check_cuda(cudaMalloc((void**)&d_A, bytes), "cudaMalloc d_A");
    check_cuda(cudaMalloc((void**)&d_C, bytes), "cudaMalloc d_C");

    /*
        复制输入数据到 GPU。
    */
    check_cuda(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice),
        "copy A host to device");

    /*
        先热身一次。

        为什么要热身？

        第一次运行 CUDA kernel 可能包含上下文初始化开销。
        热身可以让后面的计时更稳定。
    */
    contiguous_kernel << <(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> > (d_A, d_C, N);
    cudaDeviceSynchronize();

    /*
        三种访问模式测试。
    */
    run_kernel_test("1. Contiguous access: C[i] = A[i]", 0, d_A, d_C, N);

    run_kernel_test("2. Strided access: C[i] = A[i * STRIDE]", 1, d_A, d_C, N);

    run_kernel_test("3. Reverse access: C[i] = A[n - 1 - i]", 2, d_A, d_C, N);

    /*
        释放资源。
    */
    cudaFree(d_A);
    cudaFree(d_C);
    free(h_A);

    cudaDeviceReset();

    return 0;
}