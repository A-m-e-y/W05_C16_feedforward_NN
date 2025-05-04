#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>

#define INPUT_SIZE 4
#define HIDDEN_SIZE 5
#define OUTPUT_SIZE 1
#define BATCH_SIZE 1024
#define CSV_FILE "feedforward_timing.csv"

__device__ float relu(float x)
{
    return x > 0 ? x : 0;
}

__global__ void feedforward_kernel(
    float *d_input,
    float *d_w1, float *d_b1,
    float *d_w2, float *d_b2,
    float *d_output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= BATCH_SIZE)
        return;

    float hidden[HIDDEN_SIZE] = {0};

    for (int j = 0; j < HIDDEN_SIZE; j++)
    {
        float sum = 0.0f;
        for (int i = 0; i < INPUT_SIZE; i++)
        {
            sum += d_input[idx * INPUT_SIZE + i] * d_w1[i * HIDDEN_SIZE + j];
        }
        hidden[j] = relu(sum + d_b1[j]);
    }

    float sum = 0.0f;
    for (int j = 0; j < HIDDEN_SIZE; j++)
    {
        sum += hidden[j] * d_w2[j];
    }
    d_output[idx] = sum + d_b2[0];
}

void fill_random(float *arr, int size)
{
    for (int i = 0; i < size; i++)
    {
        arr[i] = ((float)rand() / RAND_MAX) * 2 - 1;
    }
}

int main()
{
    float times[5];

    for (int run = 0; run < 5; run++)
    {
        float *h_input = (float *)malloc(BATCH_SIZE * INPUT_SIZE * sizeof(float));
        float *h_w1 = (float *)malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
        float *h_b1 = (float *)malloc(HIDDEN_SIZE * sizeof(float));
        float *h_w2 = (float *)malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
        float *h_b2 = (float *)malloc(OUTPUT_SIZE * sizeof(float));
        float *h_output = (float *)malloc(BATCH_SIZE * sizeof(float));

        fill_random(h_input, BATCH_SIZE * INPUT_SIZE);
        fill_random(h_w1, INPUT_SIZE * HIDDEN_SIZE);
        fill_random(h_b1, HIDDEN_SIZE);
        fill_random(h_w2, HIDDEN_SIZE * OUTPUT_SIZE);
        fill_random(h_b2, OUTPUT_SIZE);

        float *d_input, *d_w1, *d_b1, *d_w2, *d_b2, *d_output;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        cudaMalloc(&d_input, BATCH_SIZE * INPUT_SIZE * sizeof(float));
        cudaMalloc(&d_w1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
        cudaMalloc(&d_b1, HIDDEN_SIZE * sizeof(float));
        cudaMalloc(&d_w2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
        cudaMalloc(&d_b2, OUTPUT_SIZE * sizeof(float));
        cudaMalloc(&d_output, BATCH_SIZE * sizeof(float));

        cudaMemcpy(d_input, h_input, BATCH_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_w1, h_w1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b1, h_b1, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_w2, h_w2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b2, h_b2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (BATCH_SIZE + threads - 1) / threads;
        feedforward_kernel<<<blocks, threads>>>(d_input, d_w1, d_b1, d_w2, d_b2, d_output);

        cudaMemcpy(h_output, d_output, BATCH_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_w1);
        cudaFree(d_b1);
        cudaFree(d_w2);
        cudaFree(d_b2);
        cudaFree(d_output);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        times[run] = ms;
        printf("Run %d: %.4f ms\n", run + 1, ms);

        free(h_input);
        free(h_w1);
        free(h_b1);
        free(h_w2);
        free(h_b2);
        free(h_output);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    float avg = (times[2] + times[3] + times[4]) / 3.0f;
    printf("Feedforward CUDA (Total) avg time (runs 3–5): %.4f ms\n", avg);

    FILE *fp = fopen(CSV_FILE, "w");
    fprintf(fp, "backend,batch_size,avg_time_ms\n");
    fprintf(fp, "cuda,%d,%.4f\n", BATCH_SIZE, avg);
    fclose(fp);

    return 0;
}
