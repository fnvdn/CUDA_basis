#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>

// 矩阵大小：4 x 4
// 为了方便观察输出，这里用很小的矩阵。
#define N 4

// ===============================
// GPU kernel 1：矩阵加法
// ===============================
//
// 每个 GPU 线程负责计算矩阵 C_add 中的一个元素：
//
// C_add[row][col] = A[row][col] + B[row][col]
//
// A、B、C 都是按一维数组存储的二维矩阵。
// 对于 N x N 矩阵，元素 [row][col] 在线性数组中的位置是：
//
// index = row * N + col
//
__global__ void matrix_add(const int* A, const int* B, int* C_add)
{
    // 计算当前线程负责的列号 col。
    //
    // blockIdx.x：当前 block 在 x 方向，也就是列方向的编号。
    // blockDim.x：每个 block 在 x 方向有多少线程。
    // threadIdx.x：当前线程在 block 内 x 方向的编号。
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 计算当前线程负责的行号 row。
    //
    // blockIdx.y：当前 block 在 y 方向，也就是行方向的编号。
    // blockDim.y：每个 block 在 y 方向有多少线程。
    // threadIdx.y：当前线程在 block 内 y 方向的编号。
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // 边界检查：
    // 如果启动的线程超过矩阵范围，就不进行计算。
    if (row < N && col < N) {
        int index = row * N + col;
        C_add[index] = A[index] + B[index];
    }
}

// ===============================
// GPU kernel 2：矩阵乘法
// ===============================
//
// 每个 GPU 线程负责计算矩阵 C_mul 中的一个元素：
//
// C_mul[row][col] = A[row][0] * B[0][col]
//                 + A[row][1] * B[1][col]
//                 + A[row][2] * B[2][col]
//                 + A[row][3] * B[3][col]
//
// 也就是：
// A 的第 row 行 和 B 的第 col 列 做点积。
//
__global__ void matrix_mul(const int* A, const int* B, int* C_mul)
{
    // 当前线程负责的列号。
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 当前线程负责的行号。
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // 边界检查，防止越界访问。
    if (row < N && col < N) {
        int sum = 0;

        // 矩阵乘法的核心循环。
        //
        // 对于 C[row][col]，
        // 需要 A 的第 row 行，乘以 B 的第 col 列。
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }

        C_mul[row * N + col] = sum;
    }
}

// ===============================
// CPU 辅助函数：打印矩阵
// ===============================
//
// 这个函数在 CPU 上运行。
// 用来把一维数组形式的矩阵按二维格式打印出来。
//
void print_matrix(const char* name, const int* M)
{
    printf("%s:\n", name);

    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            printf("%4d", M[row * N + col]);
        }
        printf("\n");
    }

    printf("\n");
}

int main()
{
    // 矩阵总元素数：N x N。
    const int num_elements = N * N;

    // 矩阵占用的字节数。
    const int size = num_elements * sizeof(int);

    // ===============================
    // 1. 在 CPU 端准备输入矩阵
    // ===============================
    //
    // h_ 前缀表示 host，也就是 CPU 内存。
    //
    // A =
    //  1  2  3  4
    //  5  6  7  8
    //  9 10 11 12
    // 13 14 15 16
    //
    int h_A[num_elements] = {
        1,  2,  3,  4,
        5,  6,  7,  8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };

    // B =
    // 16 15 14 13
    // 12 11 10  9
    //  8  7  6  5
    //  4  3  2  1
    //
    int h_B[num_elements] = {
        16, 15, 14, 13,
        12, 11, 10,  9,
         8,  7,  6,  5,
         4,  3,  2,  1
    };

    // CPU 端结果数组。
    int h_C_add[num_elements] = { 0 };
    int h_C_mul[num_elements] = { 0 };

    // ===============================
    // 2. 在 GPU 显存中分配空间
    // ===============================
    //
    // d_ 前缀表示 device，也就是 GPU 显存。
    //
    int* d_A = nullptr;
    int* d_B = nullptr;
    int* d_C_add = nullptr;
    int* d_C_mul = nullptr;

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C_add, size);
    cudaMalloc((void**)&d_C_mul, size);

    // ===============================
    // 3. 把输入矩阵从 CPU 复制到 GPU
    // ===============================
    //
    // cudaMemcpyHostToDevice 表示：
    // Host -> Device，也就是 CPU 内存到 GPU 显存。
    //
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // ===============================
    // 4. 设置二维线程组织
    // ===============================
    //
    // dim3 是 CUDA 提供的三维维度类型。
    //
    // block(2, 2) 表示：
    // 每个 block 有 2 x 2 = 4 个线程。
    //
    // grid(2, 2) 表示：
    // 整个 grid 有 2 x 2 = 4 个 block。
    //
    // 所以总线程数为：
    // 4 个 block * 每个 block 4 个线程 = 16 个线程。
    //
    // 正好对应 4 x 4 矩阵的 16 个元素。
    //
    dim3 block(2, 2);
    dim3 grid((N + block.x - 1) / block.x,
        (N + block.y - 1) / block.y);

    // ===============================
    // 5. 启动 GPU kernel：矩阵加法
    // ===============================
    matrix_add << <grid, block >> > (d_A, d_B, d_C_add);

    cudaError_t add_status = cudaDeviceSynchronize();
    if (add_status != cudaSuccess) {
        std::fprintf(stderr, "matrix_add CUDA error: %s\n",
            cudaGetErrorString(add_status));
        return 1;
    }

    // ===============================
    // 6. 启动 GPU kernel：矩阵乘法
    // ===============================
    matrix_mul << <grid, block >> > (d_A, d_B, d_C_mul);

    cudaError_t mul_status = cudaDeviceSynchronize();
    if (mul_status != cudaSuccess) {
        std::fprintf(stderr, "matrix_mul CUDA error: %s\n",
            cudaGetErrorString(mul_status));
        return 1;
    }

    // ===============================
    // 7. 把结果从 GPU 复制回 CPU
    // ===============================
    //
    // cudaMemcpyDeviceToHost 表示：
    // Device -> Host，也就是 GPU 显存到 CPU 内存。
    //
    cudaMemcpy(h_C_add, d_C_add, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_mul, d_C_mul, size, cudaMemcpyDeviceToHost);

    // ===============================
    // 8. 在 CPU 端打印结果
    // ===============================
    print_matrix("Matrix A", h_A);
    print_matrix("Matrix B", h_B);
    print_matrix("A + B", h_C_add);
    print_matrix("A * B", h_C_mul);

    // ===============================
    // 9. 释放 GPU 显存
    // ===============================
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_add);
    cudaFree(d_C_mul);

    // 重置 GPU 设备。
    cudaDeviceReset();

    return 0;
}