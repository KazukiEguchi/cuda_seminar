#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <random>

using namespace std;

struct Atoms{
  int N;
  double *x,*y,*vx,*vy;
  double *d_x,*d_y,*d_vx,*d_vy;
  //乱数の状態
  curandState *random_fx,*random_fy;

  Atoms(int _N){
    N = _N;
    //allocate host memory
    x = (double*)malloc(size);
    y = (double*)malloc(size);
    vx = (double*)malloc(size);
    vy= (double*)malloc(size);

    //allocate device memory
    cudaMalloc(&d_x,size);
    cudaMalloc(&d_y,size);
    cudaMalloc(&d_vx,size);
    cudaMalloc(&d_vy,size);
    cudaMalloc(&random_fx,N * sizeof(curandState));
    cudaMalloc(&random_fy,N * sizeof(curandState));
  }

  ~Atoms(){
    //free host memory
    free(x);free(y);free(vx);free(vy);
    //free device memory
    cudaFree(d_x);cudaFree(d_y);cudaFree(d_vx);cudaFree(d_vy);cudaFree(random_fx);cudaFree(random_fy);
  }

  __global__ void Velocity_conf_zero(){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N){
      d_vx[idx] = 0.0;
      d_vy[idx] = 0.0;
      d_x[idx] = 0.0;
      d_y[idx] = 0.0;
    }
  }

  /*__global__ void Initial_conf(double L){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    random_device seed;
    curandState *random_x,*random_y;
    cudaMalloc(&random_fx,N * sizeof(curandState));
    cudaMalloc(&random_fy,N * sizeof(curandState));
    setCurand<<<blocks,threads>>>(seed(), random_x);
    setCurand<<<blocks,threads>>>(seed(), random_y);

    if(idx < N){
      d_x[idx] = L * curand_uniform_double(&random_x[idx]);
      d_y[idx] = L * curand_uniform_double(&random_y[idx]);
    }

    cudaFree(random_x);cudaFree(random_y);
  }*/
};
