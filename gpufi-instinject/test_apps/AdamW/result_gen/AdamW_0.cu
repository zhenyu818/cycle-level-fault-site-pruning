#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define RANDOM_SEED 12345

void cudaCheck(cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}

__device__ inline float lerp(float start, float end, float weight) {
    return fma(weight, end, fma(-weight, start, start));
}

__global__ void adamw_kernel2(float *params_memory, const float *grads_memory, float *m_memory, float *v_memory,
                              long num_parameters, float learning_rate, float beta1, float beta2,
                              float beta1_correction, float beta2_correction, float eps, float weight_decay) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_parameters)
        return;
    float grad = grads_memory[i];
    float m = m_memory[i];
    float v = v_memory[i];
    // update the first moment (momentum)
    m = lerp(grad, m, beta1);
    m_memory[i] = m;
    // update the second moment (RMSprop)
    v = lerp(grad * grad, v, beta2);
    v_memory[i] = v;
    m /= beta1_correction; // m_hat
    v /= beta2_correction; // v_hat
    params_memory[i] -= learning_rate * (m / (sqrtf(v) + eps) + weight_decay * params_memory[i]);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        printf("Usage: %s <num_parameters> <t>\n", argv[0]);
        printf("<num_parameters>: number of parameters (long integer, size of arrays)\n");
        printf("<t>: current time step for bias correction (int, >=1)\n");
        return 1;
    }
    long num_parameters = atol(argv[1]);
    int t = atoi(argv[2]);
    if (num_parameters <= 0 || t < 1) {
        printf("Invalid arguments.\n");
        return 1;
    }

    const float learning_rate = 1e-3f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps = 1e-8f;
    const float weight_decay = 0.0f;
    float beta1_correction = 1.0f - powf(beta1, t);
    float beta2_correction = 1.0f - powf(beta2, t);

    srand(RANDOM_SEED);

    float *params_memory = (float *)malloc(num_parameters * sizeof(float));
    float *grads_memory = (float *)malloc(num_parameters * sizeof(float));
    float *m_memory = (float *)malloc(num_parameters * sizeof(float));
    float *v_memory = (float *)malloc(num_parameters * sizeof(float));
    for (long i = 0; i < num_parameters; i++) {
        float r = (float)rand() / RAND_MAX;
        params_memory[i] = r * 2.0f - 1.0f; // uniform [-1, 1]
        grads_memory[i] = r * 2.0f - 1.0f;  // uniform [-1, 1]
        m_memory[i] = r * 2.0f - 1.0f;      // uniform [-1, 1]
        v_memory[i] = r;                    // uniform [0, 1]
    }

    float *d_params, *d_grads, *d_m, *d_v;
    cudaCheck(cudaMalloc(&d_params, num_parameters * sizeof(float)));
    cudaCheck(cudaMalloc(&d_grads, num_parameters * sizeof(float)));
    cudaCheck(cudaMalloc(&d_m, num_parameters * sizeof(float)));
    cudaCheck(cudaMalloc(&d_v, num_parameters * sizeof(float)));

    cudaCheck(cudaMemcpy(d_params, params_memory, num_parameters * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_grads, grads_memory, num_parameters * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_m, m_memory, num_parameters * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_v, v_memory, num_parameters * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(512);
    dim3 grid((num_parameters + block.x - 1) / block.x);
    adamw_kernel2<<<grid, block>>>(d_params, d_grads, d_m, d_v, num_parameters, learning_rate, beta1, beta2,
                                   beta1_correction, beta2_correction, eps, weight_decay);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    cudaCheck(cudaMemcpy(params_memory, d_params, num_parameters * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(m_memory, d_m, num_parameters * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(v_memory, d_v, num_parameters * sizeof(float), cudaMemcpyDeviceToHost));

    bool first = true;
    for (long i = 0; i < num_parameters; i++) {
        if (!first)
            printf(" ");
        printf("%.6f", params_memory[i]);
        first = false;
    }
    for (long i = 0; i < num_parameters; i++) {
        if (!first)
            printf(" ");
        printf("%.6f", m_memory[i]);
        first = false;
    }
    for (long i = 0; i < num_parameters; i++) {
        if (!first)
            printf(" ");
        printf("%.6f", v_memory[i]);
        first = false;
    }
    printf("\n");

    free(params_memory);
    free(grads_memory);
    free(m_memory);
    free(v_memory);
    cudaCheck(cudaFree(d_params));
    cudaCheck(cudaFree(d_grads));
    cudaCheck(cudaFree(d_m));
    cudaCheck(cudaFree(d_v));

    return 0;
}