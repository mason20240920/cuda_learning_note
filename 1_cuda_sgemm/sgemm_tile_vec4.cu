#include <cuda_runtime.h>
#include <iostream>
#define A(i, j) a[(i) * n + (j)]
#define B(i, j) b[(i) * n + (j)]

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

void random_matrix(int m, int n, float *a)
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            A(i, j) = 2.0 * (float)drand48() - 1.0;
}

/**
 * CPU端的推理函数
 */
void cpu_gemm(const float *a_ptr,
              const float *b_ptr,
              float *c_ptr,
              const int M,
              const int N,
              const int K)
{
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
        {
            float temp = 0.f;
            for (int k = 0; k < K; k++)
            {
                temp += a_ptr[m * K + k] * b_ptr[k * N + n];
            }
            c_ptr[m * N + n] = temp;
        }
}

float compare_matrices(int m, int n, float *a, float *b)
{
    float max_diff = 0.0, diff;
    int printed = 0;

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
        {
            diff = abs(A(i, j) - B(i, j));
            max_diff = (diff > max_diff ? diff : max_diff);
            if (0 == printed)
                if (max_diff > 0.5f)
                {
                    printf("\n error: i %d j %d diff %f got %f expect %f ", i, j, max_diff, A(i, j), B(i, j));
                    printed = 1;
                }
        }

    return max_diff;
}

/**
 * SGEMM: Block Tile + Thread Tile + K Tile + Vec4, with smem
 * BK:TILE_K=8 BM=BN=128
 * TM=TN=8 增加计算密度 BM/TM=16 BN/TN=16
 * dim3 blockDim(BN/TN, BM/TM);
 * dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM)
 */
__global__ void sgemm_thread_tile_vec4(float *a_ptr,
                                       float *b_ptr,
                                       float *c_ptr,
                                       const int M,
                                       const int N,
                                       const int K)
{
    // 1. 定义前缀参数
    constexpr int M_PER_BLOCK = 32;
    constexpr int N_PER_BLOCK = 32;
    constexpr int K_PER_BLOCK = 32;
    constexpr int NUM_PER_THREAD = 4;

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;

    // 计算全局坐标
    int row = by * M_PER_BLOCK + ty;
    int col = bx * N_PER_BLOCK + tx * NUM_PER_THREAD;

    float sum[NUM_PER_THREAD] = {0.0f};

    // 2. 定义shared memory
    __shared__ float s_a[M_PER_BLOCK][K_PER_BLOCK];
    __shared__ float s_b[K_PER_BLOCK][N_PER_BLOCK];

    for (int step = 0; step < K; step += K_PER_BLOCK)
    {
        // 1. 加载A的子块到共享内存
        FLOAT4(s_a[ty][tx * NUM_PER_THREAD]) = FLOAT4(a_ptr[row * K + (tx * NUM_PER_THREAD + step)]);
        // 2. 加载B的子块到共享内存
        FLOAT4(s_b[ty][tx * NUM_PER_THREAD]) = FLOAT4(b_ptr[(ty + step) * N + col]);

        __syncthreads();

// 计算子块乘积
#pragma unroll
        for (int i = 0; i < NUM_PER_THREAD; i++)
            for (int k = 0; k < K_PER_BLOCK; k++)
                sum[i] += s_a[ty][k] * s_b[k][tx * NUM_PER_THREAD + i];
        __syncthreads();
    }

    for (int i = 0; i < NUM_PER_THREAD; i++)
        c_ptr[row * N + col + i] = sum[i];
}

int main()
{
    int m = 512;
    int n = 256;
    int k = 128;
    // 获取矩阵的大小
    size_t a_matrix_mem = m * k * sizeof(float);
    size_t b_matrix_mem = k * n * sizeof(float);
    size_t c_matrix_mem = m * n * sizeof(float);
    // 分配CPU主机上的内存大小
    float *host_a = static_cast<float *>(malloc(a_matrix_mem));
    float *host_b = static_cast<float *>(malloc(b_matrix_mem));
    float *host_c = static_cast<float *>(malloc(c_matrix_mem));
    float *gpu_c = static_cast<float *>(malloc(c_matrix_mem));
    // 自动初始化
    random_matrix(m, k, host_a);
    random_matrix(k, n, host_b);

    // 定义block大小
    constexpr int BLOCK_SIZE = 32;

    // cpu端先计算一遍, 然后基于计算结果进行比较
    cpu_gemm(host_a, host_b, host_c, m, n, k);

    // 分配GPU主机上的内存大小
    float *device_a, *device_b, *device_c;
    cudaMalloc((void **)&device_a, a_matrix_mem);
    cudaMalloc((void **)&device_b, b_matrix_mem);
    cudaMalloc((void **)&device_c, c_matrix_mem);

    // 进行内存拷贝
    cudaMemcpy(device_a, host_a, a_matrix_mem, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b, b_matrix_mem, cudaMemcpyHostToDevice);

    dim3 block(8, BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);

    sgemm_thread_tile_vec4<<<grid, block>>>(device_a, device_b, device_c, m, n, k);

    // 拷贝返回内存
    cudaMemcpy(gpu_c, device_c, c_matrix_mem, cudaMemcpyDeviceToHost);

    float diff = compare_matrices(n, m, gpu_c, host_c);
    if (diff > 0.5f)
    {
        printf("diff too big !\n");
        return EXIT_FAILURE;
    }
    printf("Right\n");

    return EXIT_SUCCESS;
}