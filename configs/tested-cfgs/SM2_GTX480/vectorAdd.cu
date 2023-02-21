/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <helper_cuda.h>
/**
 * CUDA Kernel Device code
 */
__global__ void vectorAdd(int *data_in, int *data_out, int numElements) {
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

/**
 * Host main routine
 */
int main(void) {
  // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;

  // Print the vector length to be used, and compute its size
  int numElements = 32;
  size_t size = numElements * sizeof(int);
  printf("[Vector addition of %d elements]\n", numElements);

  int *h_data_in = (int *)malloc(size);
  int *h_data_out = (int *)malloc(size);

  // Initialize the host input vectors
  for (int i = 0; i < numElements; ++i) {
    h_data_in[i] = i;
  }

  int *d_data_in = NULL;
  err = cudaMalloc((void **)&d_data_in, size);
  int *d_data_out = NULL;
  err = cudaMalloc((void **)&d_data_out, size);

  printf("Copy input data from the host memory to the CUDA device\n");
  err = cudaMemcpy(d_data_in, h_data_in, size, cudaMemcpyHostToDevice);

  // Launch the Vector Add CUDA Kernel
  int threadsPerBlock = 32;
  int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
         threadsPerBlock);
  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_data_in, d_data_out, numElements);
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

  // Free device global memory
  err = cudaFree(d_data_in);
  err = cudaFree(d_data_out);

  // Free host memory
  free(h_data_in);
  free(h_data_out);

  printf("Done\n");
  return 0;
}
