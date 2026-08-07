#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

/*
    矩阵大小 N = 4

    本实验使用 4 x 4 矩阵。
    选择 4 x 4 是为了方便打印和手算检查。
*/
#define N 4

/*
    tile 大小 TILE_SIZE = 2

    一个 tile 就是从大矩阵中切出来的一小块。

    本实验中：
    一个 block 负责计算 C 矩阵中的一个 2 x 2 小块。
*/
#define TILE_SIZE 2

/*
    CUDA kernel：使用 shared memory 的分块矩阵乘法

    计算内容：

        C = A * B

    矩阵乘法公式：

        C[row][col]
        = A[row][0] * B[0][col]
        + A[row][1] * B[1][col]
        + A[row][2] * B[2][col]
        + A[row][3] * B[3][col]

    核心思想：

        不让每个线程都反复直接访问 global memory。

        而是让一个 block 内的线程先合作：
        1. 从 global memory 中搬一小块 A 到 shared memory；
        2. 从 global memory 中搬一小块 B 到 shared memory；
        3. 然后 block 内的线程反复使用 shared memory 中的数据做乘加。

    这样可以减少对 global memory 的访问次数。
*/
__global__ void matmul_tiled(const int* A, const int* B, int* C)
{
    /*
        As 是 A 矩阵在 shared memory 中的小块。

        注意：
        shared memory 是 block 内共享的。
        同一个 block 里的线程都可以访问 As。

        但是不同 block 的 As 是彼此独立的。
        block 0 的 As 和 block 1 的 As 不是同一个数组。

        TILE_SIZE = 2，所以 As 的形状是：

            As[0][0]  As[0][1]
            As[1][0]  As[1][1]
    */
    __shared__ int As[TILE_SIZE][TILE_SIZE];

    /*
        Bs 是 B 矩阵在 shared memory 中的小块。

        它和 As 类似，也是每个 block 自己拥有一份。
    */
    __shared__ int Bs[TILE_SIZE][TILE_SIZE];

    /*
        threadIdx.x：
        当前线程在 block 内的列编号。

        threadIdx.y：
        当前线程在 block 内的行编号。

        因为一个 block 是 2 x 2 个线程，
        所以 block 内线程的位置是：

            (threadIdx.y, threadIdx.x)

            (0, 0)   (0, 1)
            (1, 0)   (1, 1)

        每个线程负责计算 C 矩阵中的一个元素。
    */
    int local_col = threadIdx.x;
    int local_row = threadIdx.y;

    /*
        blockIdx.x：
        当前 block 在 grid 中的列编号。

        blockIdx.y：
        当前 block 在 grid 中的行编号。

        一个 block 计算 C 的一个 2 x 2 tile。

        所以当前线程负责的 C 元素坐标是：

            row = blockIdx.y * TILE_SIZE + local_row
            col = blockIdx.x * TILE_SIZE + local_col

        举例：

        blockIdx = (0, 0) 时：
            这个 block 计算 C 左上角 2 x 2 区域：

            C[0][0]  C[0][1]
            C[1][0]  C[1][1]

        blockIdx = (1, 0) 时：
            这个 block 计算 C 右上角 2 x 2 区域：

            C[0][2]  C[0][3]
            C[1][2]  C[1][3]
    */
    int row = blockIdx.y * TILE_SIZE + local_row;
    int col = blockIdx.x * TILE_SIZE + local_col;

    /*
        sum 用来保存当前线程计算的结果。

        每个线程都有自己的 sum。
        它是线程私有变量，通常放在寄存器中。
    */
    int sum = 0;

    /*
        矩阵乘法要沿着 k 方向做累加。

        因为 N = 4，TILE_SIZE = 2，
        所以 k 方向被分成两个阶段：

            phase = 0：处理 k = 0, 1
            phase = 1：处理 k = 2, 3

        每个阶段中：
            1. 搬 A 的一个 2 x 2 小块到 As；
            2. 搬 B 的一个 2 x 2 小块到 Bs；
            3. 用 As 和 Bs 做一部分乘加。
    */
    for (int phase = 0; phase < N / TILE_SIZE; phase++) {

        /*
            当前线程从 A 中读取一个元素，放入 shared memory 的 As。

            A 是二维矩阵，但在内存中按一维数组存储。

            二维坐标：
                A[row][col]

            对应的一维下标：
                row * N + col

            这里读取的是：

                A[row][phase * TILE_SIZE + local_col]

            写入的是：

                As[local_row][local_col]

            也就是说：
            每个线程按照自己在 block 内的位置，
            搬运 A tile 中对应位置的一个元素。
        */
        As[local_row][local_col] =
            A[row * N + (phase * TILE_SIZE + local_col)];

        /*
            当前线程从 B 中读取一个元素，放入 shared memory 的 Bs。

            这里读取的是：

                B[phase * TILE_SIZE + local_row][col]

            写入的是：

                Bs[local_row][local_col]

            为什么 A 和 B 的读取公式不一样？

            因为计算 C[row][col] 时需要：

                A[row][k] 和 B[k][col]

            A 固定 row，沿着列方向取；
            B 固定 col，沿着行方向取。
        */
        Bs[local_row][local_col] =
            B[(phase * TILE_SIZE + local_row) * N + col];

        /*
            第一次同步。

            作用：
            保证 block 内所有线程都已经把 A tile 和 B tile
            搬进 shared memory。

            如果没有这句，可能出现：
            某些线程已经开始用 As、Bs 计算，
            但另一些线程还没把数据搬进来。

            这样会读到错误数据。
        */
        __syncthreads();

        /*
            使用 shared memory 中的 As 和 Bs 做部分乘加。

            因为 TILE_SIZE = 2，所以每个 phase 做两次乘加：

                sum += As[local_row][0] * Bs[0][local_col];
                sum += As[local_row][1] * Bs[1][local_col];

            举例：
            假设当前线程负责 C[0][0]。

            phase = 0 时：
                sum += A[0][0] * B[0][0];
                sum += A[0][1] * B[1][0];

            phase = 1 时：
                sum += A[0][2] * B[2][0];
                sum += A[0][3] * B[3][0];

            两个 phase 做完后，sum 就是 C[0][0]。
        */
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[local_row][k] * Bs[k][local_col];
        }

        /*
            第二次同步。

            作用：
            保证所有线程都已经使用完当前 phase 的 As 和 Bs。

            为什么需要？

            因为下一轮 phase 会覆盖 As 和 Bs。
            如果某些线程还没用完，另一些线程已经开始写入下一块数据，
            就会破坏 shared memory 中的数据。
        */
        __syncthreads();
    }

    /*
        所有 phase 结束后，sum 就是当前线程负责的 C[row][col]。

        把结果写回 global memory。
    */
    C[row * N + col] = sum;
}

/*
    打印矩阵的辅助函数。

    虽然矩阵在逻辑上是二维的，
    但它在内存中是按一维数组存储的。

    M[row][col] 对应：
        M[row * N + col]
*/
void print_matrix(const char* name, const int* M)
{
    printf("%s:\n", name);

    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            printf("%4d ", M[row * N + col]);
        }
        printf("\n");
    }

    printf("\n");
}

int main()
{
    /*
        CPU 端矩阵 A。

        A =

            1   2   3   4
            5   6   7   8
            9  10  11  12
           13  14  15  16
    */
    int h_A[N * N] = {
         1,  2,  3,  4,
         5,  6,  7,  8,
         9, 10, 11, 12,
        13, 14, 15, 16
    };

    /*
        CPU 端矩阵 B。

        B =

            1  0  0  1
            0  1  1  0
            1  0  0  1
            0  1  1  0

        这个 B 不是单位矩阵，
        但它的数值比较简单，方便观察结果。
    */
    int h_B[N * N] = {
        1, 0, 0, 1,
        0, 1, 1, 0,
        1, 0, 0, 1,
        0, 1, 1, 0
    };

    /*
        CPU 端矩阵 C，用来接收 GPU 计算结果。
    */
    int h_C[N * N] = { 0 };

    /*
        GPU 端指针。

        d_A、d_B、d_C 都指向 GPU 的 global memory。
    */
    int* d_A = nullptr;
    int* d_B = nullptr;
    int* d_C = nullptr;

    /*
        在 GPU global memory 中分配 A、B、C 的空间。
    */
    cudaMalloc((void**)&d_A, N * N * sizeof(int));
    cudaMalloc((void**)&d_B, N * N * sizeof(int));
    cudaMalloc((void**)&d_C, N * N * sizeof(int));

    /*
        把 A 和 B 从 CPU 内存复制到 GPU 内存。

        cudaMemcpyHostToDevice 表示：
        Host，也就是 CPU -> Device，也就是 GPU。
    */
    cudaMemcpy(d_A, h_A, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(int), cudaMemcpyHostToDevice);

    /*
        一个 block 中有 2 x 2 个线程。

        这 4 个线程共同计算 C 中的一个 2 x 2 小块。
    */
    dim3 threads_per_block(TILE_SIZE, TILE_SIZE);

    /*
        grid 中有 2 x 2 个 block。

        因为：
            C 是 4 x 4
            每个 block 计算 2 x 2

        所以需要：
            横向 2 个 block
            纵向 2 个 block
    */
    dim3 blocks_per_grid(N / TILE_SIZE, N / TILE_SIZE);

    /*
        启动 kernel。

        <<<blocks_per_grid, threads_per_block>>>

        表示：
            启动 2 x 2 个 block；
            每个 block 有 2 x 2 个线程。
    */
    matmul_tiled << <blocks_per_grid, threads_per_block >> > (d_A, d_B, d_C);

    /*
        等待 GPU 执行完成，并检查 kernel 是否出错。

        CUDA kernel 启动后，CPU 不会自动等待它执行完。
        所以这里需要 cudaDeviceSynchronize()。
    */
    cudaError_t status = cudaDeviceSynchronize();

    if (status != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(status));

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        return 1;
    }

    /*
        把 C 从 GPU 内存复制回 CPU 内存。

        cudaMemcpyDeviceToHost 表示：
        Device，也就是 GPU -> Host，也就是 CPU。
    */
    cudaMemcpy(h_C, d_C, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    /*
        打印 A、B、C。
    */
    print_matrix("A", h_A);
    print_matrix("B", h_B);
    print_matrix("C = A * B", h_C);

    /*
        释放 GPU 内存。
    */
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}