# CUDA_basis
a few experiment about CUDA basis

01_thread_index	线程编号打印	打印 blockIdx、threadIdx、blockDim，理解线程如何编号

02_vector_add	向量加法	一个线程处理一个数组元素，完成 C[i] = A[i] + B[i]

03_matrix_add_mul	矩阵加法与乘法	用二维 thread/block 索引计算矩阵加法和普通矩阵乘法

04_atomic_counter	原子计数器	多个线程同时更新同一个计数器，理解 atomicAdd

05_shared_memory_tile	shared memory 数据搬运	把数据从 global memory 搬到 shared memory，并做 block 内反转

06_shared_memory_matmul	shared memory 分块矩阵乘法	用 shared memory 复用矩阵小块，减少 global memory 访问

07_stream_event_overlap	流与事件	用 stream 实现数据传输和 kernel 计算的异步/重叠，用 event 计时

08_vector_add_timing	CPU vs GPU 计时	比较 CPU 和 GPU 做向量加法的耗时，理解数据规模和传输开销

09_memory_access_pattern	访存模式实验	比较连续访存和非连续访存，理解 global memory 合并访问和性能差异


01 到 04 主要巩固第二章的 CUDA 编程模型：线程组织、数据索引、kernel、原子操作。

05 到 09 开始进入第四章相关内容：shared memory、global memory、访存效率、数据搬运、性能计时。

10 和 11 是进阶实验：

10_mini_moe_cache_sim	mini-MoE expert cache 模拟	用 Python 模拟 MoE 中 expert 的访问、缓存命中/缺失、LRU 淘汰和预取策略

11_mini_moe_mlp	小型 MLP expert 的 mini-MoE 前向	把 expert 从简单编号/矩阵换成小型 MLP，用 PyTorch 模拟一个 mini-MoE 层的前向推理
