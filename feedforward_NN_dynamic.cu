#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>

#define INPUT_SIZE 4
#define HIDDEN_SIZE 10
#define OUTPUT_SIZE 1
#define BATCH_SIZE 1024
#define REPEAT 5
#define CSV_FILE "feedforward_layer_scaling.csv"

__device__ float relu(float x)
{
    return x > 0 ? x : 0;
}

__global__ void feedforward_kernel(
    float *input, float *weights, float *biases,
    float *out_weights, float *out_bias,
    float *output,
    int num_hidden_layers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= BATCH_SIZE)
        return;

    float layer_in[HIDDEN_SIZE];
    float layer_out[HIDDEN_SIZE];

    // First layer: input -> hidden[0]
    for (int j = 0; j < HIDDEN_SIZE; j++)
    {
        float sum = 0.0f;
        for (int i = 0; i < INPUT_SIZE; i++)
        {
            sum += input[idx * INPUT_SIZE + i] * weights[i * HIDDEN_SIZE + j];
        }
        layer_out[j] = relu(sum + biases[j]);
    }

    // Hidden[i-1] -> Hidden[i] for layers 1 to num_hidden_layers-1
    for (int l = 1; l < num_hidden_layers; l++)
    {
        for (int j = 0; j < HIDDEN_SIZE; j++)
        {
            float sum = 0.0f;
            for (int i = 0; i < HIDDEN_SIZE; i++)
            {
                sum += layer_out[i] * weights[(l * HIDDEN_SIZE * HIDDEN_SIZE) + i * HIDDEN_SIZE + j];
            }
            layer_in[j] = relu(sum + biases[l * HIDDEN_SIZE + j]);
        }
        for (int i = 0; i < HIDDEN_SIZE; i++)
            layer_out[i] = layer_in[i];
    }

    // Last layer: hidden -> output
    float sum = 0.0f;
    for (int i = 0; i < HIDDEN_SIZE; i++)
    {
        sum += layer_out[i] * out_weights[i];
    }
    output[idx] = sum + out_bias[0];
}

void fill_random(float *arr, int size)
{
    for (int i = 0; i < size; i++)
    {
        arr[i] = ((float)rand() / RAND_MAX) * 2 - 1;
    }
}

int main(int argc, char **argv)
{
    int max_layers = 5;
    if (argc >= 2)
        max_layers = atoi(argv[1]);

    FILE *fp = fopen(CSV_FILE, "w");
    fprintf(fp, "backend,hidden_layers,batch_size,avg_time_ms\n");

    for (int L = 1; L <= max_layers; L++)
    {
        float times[REPEAT];

        for (int run = 0; run < REPEAT; run++)
        {
            float *h_input = (float *)malloc(BATCH_SIZE * INPUT_SIZE * sizeof(float));
            float *h_weights = (float *)malloc((L * HIDDEN_SIZE * HIDDEN_SIZE + INPUT_SIZE * HIDDEN_SIZE) * sizeof(float));
            float *h_biases = (float *)malloc(L * HIDDEN_SIZE * sizeof(float));
            float *h_out_w = (float *)malloc(HIDDEN_SIZE * sizeof(float));
            float *h_out_b = (float *)malloc(sizeof(float));
            float *h_output = (float *)malloc(BATCH_SIZE * sizeof(float));

            fill_random(h_input, BATCH_SIZE * INPUT_SIZE);
            fill_random(h_weights, INPUT_SIZE * HIDDEN_SIZE);
            fill_random(h_weights + INPUT_SIZE * HIDDEN_SIZE, (L - 1) * HIDDEN_SIZE * HIDDEN_SIZE);
            fill_random(h_biases, L * HIDDEN_SIZE);
            fill_random(h_out_w, HIDDEN_SIZE);
            fill_random(h_out_b, 1);

            float *d_input, *d_weights, *d_biases, *d_out_w, *d_out_b, *d_output;
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);

            cudaMalloc(&d_input, BATCH_SIZE * INPUT_SIZE * sizeof(float));
            cudaMalloc(&d_weights, (L * HIDDEN_SIZE * HIDDEN_SIZE + INPUT_SIZE * HIDDEN_SIZE) * sizeof(float));
            cudaMalloc(&d_biases, L * HIDDEN_SIZE * sizeof(float));
            cudaMalloc(&d_out_w, HIDDEN_SIZE * sizeof(float));
            cudaMalloc(&d_out_b, sizeof(float));
            cudaMalloc(&d_output, BATCH_SIZE * sizeof(float));

            cudaMemcpy(d_input, h_input, BATCH_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_weights, h_weights, (L * HIDDEN_SIZE * HIDDEN_SIZE + INPUT_SIZE * HIDDEN_SIZE) * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_biases, h_biases, L * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_out_w, h_out_w, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_out_b, h_out_b, sizeof(float), cudaMemcpyHostToDevice);

            int threads = 256;
            int blocks = (BATCH_SIZE + threads - 1) / threads;
            feedforward_kernel<<<blocks, threads>>>(d_input, d_weights, d_biases, d_out_w, d_out_b, d_output, L);

            cudaMemcpy(h_output, d_output, BATCH_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

            cudaFree(d_input);
            cudaFree(d_weights);
            cudaFree(d_biases);
            cudaFree(d_out_w);
            cudaFree(d_out_b);
            cudaFree(d_output);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms = 0;
            cudaEventElapsedTime(&ms, start, stop);
            times[run] = ms;

            free(h_input);
            free(h_weights);
            free(h_biases);
            free(h_out_w);
            free(h_out_b);
            free(h_output);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        float avg = (times[2] + times[3] + times[4]) / 3.0f;
        printf("Hidden Layers: %d | Avg Time (runs 3-5): %.4f ms\n", L, avg);
        fprintf(fp, "cuda,%d,%d,%.4f\n", L, BATCH_SIZE, avg);
    }

    fclose(fp);
    return 0;
}
