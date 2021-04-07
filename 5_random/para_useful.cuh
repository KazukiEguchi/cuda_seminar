#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <fstream>
#include <curand.h>
#include <curand_kernel.h>

using namespace std;

//粒子数
const int N = 1024;
const int threads = 1024;
const int blocks = (N -1+threads)/threads;

random_device seed;

void copyH2D(void* dest,void* src,std::size_t size){
  cudaMemcpy(dest,src,size,cudaMemcpyHostToDevice);
}

void copyD2H(void* dest,void* src,std::size_t size){
  cudaMemcpy(dest,src,size,cudaMemcpyDeviceToHost);
}

//curandStateの初期化
__global__ void setCurand(unsigned long long seed, curandState *state){
    uint i_global = threadIdx.x + blockIdx.x*blockDim.x;
    curand_init(seed, i_global, 0, &state[i_global]);
}

//乱数の設定
void cuda_random_set(curandState *state){
  setCurand<<<blocks,threads>>>(seed(), random_x);
}
