/* Copyright (c) 2023, ZhaoSiying12138. All rights reserved. */

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void test_mpipdom_dualpath(int *data_in, int *data_out, int numElements) {
  int tid = threadIdx.x;
  if (tid < 8) {
    data_out[tid] = data_in[tid] + 10000;
  } else {
    data_out[tid] = data_in[tid] + 100;
    data_out[tid] = data_out[tid] + 100;
    data_out[tid] = data_out[tid] + 100;
    data_out[tid] = data_out[tid] + 100;
  }
  data_out[tid] += 1;
}

int main(void) {
  cudaError_t err = cudaSuccess;

  int numElements = 32;
  size_t size = numElements * sizeof(int);

  int *h_data_in = (int *)malloc(size);
  int *h_data_out = (int *)malloc(size);

  for (int i = 0; i < numElements; ++i) {
    h_data_in[i] = i;
  }

  int *d_data_in = NULL;
  err = cudaMalloc((void **)&d_data_in, size);
  int *d_data_out = NULL;
  err = cudaMalloc((void **)&d_data_out, size);

  printf("Copy input data from the host memory to the CUDA device\n");
  err = cudaMemcpy(d_data_in, h_data_in, size, cudaMemcpyHostToDevice);

  int threadsPerBlock = 32;
  int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
         threadsPerBlock);
  test_mpipdom_dualpath<<<blocksPerGrid, threadsPerBlock>>>(d_data_in, d_data_out, numElements);
  err = cudaGetLastError();

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  
  printf("Copy output data from CUDA device to the host memory\n");
  err = cudaMemcpy(h_data_out, d_data_out, size, cudaMemcpyDeviceToHost);

  for (int i = 0; i < numElements; i++)
    printf("%d ", h_data_out[i]);
  printf("\n");

  err = cudaFree(d_data_in);
  err = cudaFree(d_data_out);

  free(h_data_in);
  free(h_data_out);

  printf("Done\n");
  return 0;
}
