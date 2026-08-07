#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

/*
    实验名称：
    07_stream_event_overlap

    实验目标：
    1. 理解 CUDA stream 是什么。
    2. 理解 CUDA event 是什么。
    3. 学会用 event 测量 GPU 上一段操作的时间。
    4. 初步观察多个 stream 是否可能并发执行。

    核心概念：

    stream：
        CUDA stream 可以理解为 GPU 上的一条任务队列。
        同一个 stream 内的任务按顺序执行。
        不同 stream 内的任务在硬件资源允许时可能并发执行。

    event：
        CUDA event 可以理解为 GPU 时间线上的一个标记点。
        可以用 start event 和 stop event 测量 GPU 执行时间。
*/

#define N (1 << 20)          // 数组长度：约 100 万个 float
#define BLOCK_SIZE 256
#define REPEAT 200           // 每个线程重复计算次数，用来增加 kernel 运行时间

/*
    heavy_add_kernel

    这个 kernel 做一个稍微重一点的向量运算。

    输入：
        A, B

    输出：
        C

    每个线程处理一个元素：

        C[i] = A[i] + B[i]

    但是为了让 kernel 运行时间更明显，
    我们在每个线程里重复做 REPEAT 次简单计算。
*/
__global__ void heavy_add_kernel(const float* A, const float* B, float* C, int n)
{
    /*
        计算当前线程对应的全局编号。

        blockIdx.x：
            当前 block 在 grid 中的编号。

        blockDim.x：
            每个 block 中的线程数。

        threadIdx.x：
            当前线程在 block 内的编号。
    */
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    /*
        防止线程编号超过数组范围。

        因为 grid 的线程总数可能略大于 n。
    */
    if (i < n) {
        /*
            x 是线程私有变量，通常存放在寄存器中。
        */
        float x = A[i];
        float y = B[i];

        /*
            重复计算，故意增加 kernel 的计算量。

            这里不只是简单 C[i] = A[i] + B[i]，
            而是让每个线程多做一些加法和乘法，
            这样 event 测出来的时间更明显。
        */
        for (int r = 0; r < REPEAT; r++) {
            x = x * 1.000001f + y * 0.999999f;
            y = y * 1.000002f + 0.000001f;
        }

        /*
            把结果写回 global memory。
        */
        C[i] = x + y;
    }
}

/*
    初始化 CPU 数组。
*/
void init_array(float* data, int n, float value)
{
    for (int i = 0; i < n; i++) {
        data[i] = value;
    }
}

/*
    检查 CUDA API 是否出错的简单函数。
*/
void check_cuda(cudaError_t status, const char* message)
{
    if (status != cudaSuccess) {
        printf("CUDA error at %s: %s\n", message, cudaGetErrorString(status));
    }
}

int main()
{
    /*
        计算数组所需字节数。
    */
    int bytes = N * sizeof(float);

    /*
        CPU 端数组。

        h_A1, h_B1, h_C1 用于第一组计算。
        h_A2, h_B2, h_C2 用于第二组计算。
    */
    float* h_A1 = (float*)malloc(bytes);
    float* h_B1 = (float*)malloc(bytes);
    float* h_C1 = (float*)malloc(bytes);

    float* h_A2 = (float*)malloc(bytes);
    float* h_B2 = (float*)malloc(bytes);
    float* h_C2 = (float*)malloc(bytes);

    /*
        初始化 CPU 数据。
    */
    init_array(h_A1, N, 1.0f);
    init_array(h_B1, N, 2.0f);
    init_array(h_A2, N, 3.0f);
    init_array(h_B2, N, 4.0f);

    /*
        GPU 端数组指针。

        d_ 开头表示 device，也就是 GPU 内存。
    */
    float* d_A1 = nullptr;
    float* d_B1 = nullptr;
    float* d_C1 = nullptr;

    float* d_A2 = nullptr;
    float* d_B2 = nullptr;
    float* d_C2 = nullptr;

    /*
        在 GPU global memory 中分配空间。
    */
    check_cuda(cudaMalloc((void**)&d_A1, bytes), "cudaMalloc d_A1");
    check_cuda(cudaMalloc((void**)&d_B1, bytes), "cudaMalloc d_B1");
    check_cuda(cudaMalloc((void**)&d_C1, bytes), "cudaMalloc d_C1");

    check_cuda(cudaMalloc((void**)&d_A2, bytes), "cudaMalloc d_A2");
    check_cuda(cudaMalloc((void**)&d_B2, bytes), "cudaMalloc d_B2");
    check_cuda(cudaMalloc((void**)&d_C2, bytes), "cudaMalloc d_C2");

    /*
        把 CPU 数据复制到 GPU。

        这里先用普通 cudaMemcpy。
        这部分不是本实验重点。
    */
    check_cuda(cudaMemcpy(d_A1, h_A1, bytes, cudaMemcpyHostToDevice), "copy A1");
    check_cuda(cudaMemcpy(d_B1, h_B1, bytes, cudaMemcpyHostToDevice), "copy B1");

    check_cuda(cudaMemcpy(d_A2, h_A2, bytes, cudaMemcpyHostToDevice), "copy A2");
    check_cuda(cudaMemcpy(d_B2, h_B2, bytes, cudaMemcpyHostToDevice), "copy B2");

    /*
        kernel 启动配置。

        每个 block 256 个线程。
        block 数量向上取整，保证至少覆盖 N 个元素。
    */
    int threads_per_block = BLOCK_SIZE;
    int blocks_per_grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    /*
        ============================================================
        第一部分：默认流顺序执行
        ============================================================

        默认流也叫 default stream。
        如果不指定 stream，kernel 就会进入默认流。

        同一个 stream 里的 kernel 是顺序执行的：

            kernel 1 完成后，kernel 2 才开始。
    */

    cudaEvent_t start_default;
    cudaEvent_t stop_default;

    check_cuda(cudaEventCreate(&start_default), "create start_default");
    check_cuda(cudaEventCreate(&stop_default), "create stop_default");

    /*
        在 GPU 时间线上记录起点。
    */
    check_cuda(cudaEventRecord(start_default), "record start_default");

    /*
        两个 kernel 都没有指定 stream。
        所以它们都进入默认流，通常按顺序执行。
    */
    heavy_add_kernel << <blocks_per_grid, threads_per_block >> > (d_A1, d_B1, d_C1, N);
    heavy_add_kernel << <blocks_per_grid, threads_per_block >> > (d_A2, d_B2, d_C2, N);

    /*
        在 GPU 时间线上记录终点。
    */
    check_cuda(cudaEventRecord(stop_default), "record stop_default");

    /*
        等待 stop_default 对应的位置执行完成。
    */
    check_cuda(cudaEventSynchronize(stop_default), "sync stop_default");

    /*
        计算 start_default 到 stop_default 之间的 GPU 时间。
        单位是毫秒 ms。
    */
    float time_default = 0.0f;
    check_cuda(cudaEventElapsedTime(&time_default, start_default, stop_default),
        "elapsed default");

    /*
        ============================================================
        第二部分：两个不同 stream 执行
        ============================================================

        我们创建 stream1 和 stream2。

        kernel 1 放进 stream1。
        kernel 2 放进 stream2。

        不同 stream 中的任务，在硬件资源允许时可能重叠执行。
    */

    cudaStream_t stream1;
    cudaStream_t stream2;

    check_cuda(cudaStreamCreate(&stream1), "create stream1");
    check_cuda(cudaStreamCreate(&stream2), "create stream2");

    cudaEvent_t start_streams;
    cudaEvent_t stop_streams;

    check_cuda(cudaEventCreate(&start_streams), "create start_streams");
    check_cuda(cudaEventCreate(&stop_streams), "create stop_streams");

    /*
        记录多 stream 实验的起点。

        这里 event 记录在默认流中。
        对于入门实验来说，重点是看整体耗时。
    */
    check_cuda(cudaEventRecord(start_streams), "record start_streams");

    /*
        第三个参数 0 表示 shared memory 动态分配大小为 0。

        第四个参数 stream1 / stream2 表示：
        这个 kernel 放到哪个 stream 里执行。

        语法：

            kernel<<<grid, block, shared_memory_bytes, stream>>>(...);
    */
    heavy_add_kernel << <blocks_per_grid, threads_per_block, 0, stream1 >> >
        (d_A1, d_B1, d_C1, N);

    heavy_add_kernel << <blocks_per_grid, threads_per_block, 0, stream2 >> >
        (d_A2, d_B2, d_C2, N);

    /*
        等待 stream1 和 stream2 中的任务都完成。

        注意：
        cudaStreamSynchronize(stream1) 只等待 stream1。
        cudaStreamSynchronize(stream2) 只等待 stream2。
    */
    check_cuda(cudaStreamSynchronize(stream1), "sync stream1");
    check_cuda(cudaStreamSynchronize(stream2), "sync stream2");

    /*
        两个 stream 都完成后，记录终点。
    */
    check_cuda(cudaEventRecord(stop_streams), "record stop_streams");
    check_cuda(cudaEventSynchronize(stop_streams), "sync stop_streams");

    float time_streams = 0.0f;
    check_cuda(cudaEventElapsedTime(&time_streams, start_streams, stop_streams),
        "elapsed streams");

    /*
        把结果复制回 CPU。

        这里只检查少量结果，证明 kernel 确实执行了。
    */
    check_cuda(cudaMemcpy(h_C1, d_C1, bytes, cudaMemcpyDeviceToHost), "copy C1");
    check_cuda(cudaMemcpy(h_C2, d_C2, bytes, cudaMemcpyDeviceToHost), "copy C2");

    /*
        打印实验结果。

        time_default:
            两个 kernel 在默认流中顺序执行的大致耗时。

        time_streams:
            两个 kernel 放到不同 stream 后的大致耗时。

        如果 time_streams 明显小于 time_default，
        说明两个 stream 的执行可能发生了一定重叠。

        如果差不多，也不是错误。
        可能原因：
            1. kernel 本身已经占满 GPU；
            2. Debug 模式影响；
            3. GPU 调度策略；
            4. Windows WDDM 模式影响；
            5. 任务规模不适合观察 overlap。
    */
    printf("Default stream time: %.3f ms\n", time_default);
    printf("Two streams time:    %.3f ms\n", time_streams);

    printf("\nSample output:\n");
    printf("h_C1[0] = %.3f\n", h_C1[0]);
    printf("h_C2[0] = %.3f\n", h_C2[0]);

    /*
        释放 event。
    */
    cudaEventDestroy(start_default);
    cudaEventDestroy(stop_default);
    cudaEventDestroy(start_streams);
    cudaEventDestroy(stop_streams);

    /*
        释放 stream。
    */
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    /*
        释放 GPU 内存。
    */
    cudaFree(d_A1);
    cudaFree(d_B1);
    cudaFree(d_C1);

    cudaFree(d_A2);
    cudaFree(d_B2);
    cudaFree(d_C2);

    /*
        释放 CPU 内存。
    */
    free(h_A1);
    free(h_B1);
    free(h_C1);

    free(h_A2);
    free(h_B2);
    free(h_C2);

    return 0;
}