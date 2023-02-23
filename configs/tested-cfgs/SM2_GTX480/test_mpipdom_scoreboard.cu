/* Copyright (c) 2023, ZhaoSiying12138. All rights reserved. */

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void test_mpipdom_dualpath(int *data_in1, int *data_in2, int *data_out, int numElements) {
  int tid = threadIdx.x;
  int idx = data_in1[tid];
  int tmp;
  if (tid < 8) {
    tmp = data_in2[idx];
  } else {
    tmp = data_in2[tid];
    tmp = tmp + idx;
  }
  data_out[tid] = tmp;
}

int main(void) {
  cudaError_t err = cudaSuccess;

  int numElements = 32;
  size_t size = numElements * sizeof(int);

  int *h_data_in1 = (int *)malloc(size);
  int *h_data_in2 = (int *)malloc(size * 2);
  int *h_data_out = (int *)malloc(size);

  for (int i = 0; i < numElements; ++i) {
    h_data_in1[i] = i + 32;
  }
  for (int i = 0; i < numElements * 2; ++i) {
    h_data_in2[i] = i + 10000;
  }

  int *d_data_in1 = NULL;
  err = cudaMalloc((void **)&d_data_in1, size);
  int *d_data_in2 = NULL;
  err = cudaMalloc((void **)&d_data_in2, size * 2);
  int *d_data_out = NULL;
  err = cudaMalloc((void **)&d_data_out, size);

  printf("Copy input data from the host memory to the CUDA device\n");
  err = cudaMemcpy(d_data_in1, h_data_in1, size, cudaMemcpyHostToDevice);
  err = cudaMemcpy(d_data_in2, h_data_in2, size * 2, cudaMemcpyHostToDevice);

  int threadsPerBlock = 32;
  int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
         threadsPerBlock);
  test_mpipdom_dualpath<<<blocksPerGrid, threadsPerBlock>>>(d_data_in1, d_data_in2, d_data_out, numElements);
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

  err = cudaFree(d_data_in1);
  err = cudaFree(d_data_in2);
  err = cudaFree(d_data_out);

  free(h_data_in1);
  free(h_data_in2);
  free(h_data_out);

  printf("Done\n");
  return 0;
}
