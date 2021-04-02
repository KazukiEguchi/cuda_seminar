#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <fstream>
#include <curand.h>
#include <curand_kernel.h>

using namespace std;

//curandStateの初期化
__global__ void setCurand(unsigned long long seed, curandState *state){
    uint i_global = threadIdx.x + blockIdx.x*blockDim.x;
    curand_init(seed, i_global, 0, &state[i_global]);
}

void copy(double *x,double *y,int N){
  for(int i = 0;i < N;i++){
    y[i] = x[i];
  }
}

__global__ void copy_ker(double *x,double *y,int N){
  uint idx = threadIdx.x + blockIdx.x*blockDim.x;
  if(idx < N){
    y[idx] = x[idx];
  }
}

struct Atoms{
  int N;
  double *x,*y,*vx,*vy;
  double *d_x,*d_y,*d_vx,*d_vy;
  //乱数の状態
  curandState *random_fx,*random_fy;

  Atoms(int _N){
    N = _N;
    int size = N * sizeof(double);

    //シード値
    random_device seed_gen;

    int threads = 1;
    int blocks = 1;

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

    //シードを使ってcurandStateを初期化
    setCurand<<<blocks,threads>>>(seed_gen(),random_fx);
    //シードを使ってcurandStateを初期化
    setCurand<<<blocks,threads>>>(seed_gen(),random_fy);
  }

  //コピーコンストラクタ
  Atoms(const Atoms &_atoms){
    int threads = 1;
    int blocks = 1;

    N = _atoms.N;
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

    copy(_atoms.x,x,N);
    copy(_atoms.y,y,N);
    copy(_atoms.vx,vx,N);
    copy(_atoms.vy,vy,N);
    copy_ker<<<blocks,threads>>>(_atoms.d_x,d_x,N);
    copy_ker<<<blocks,threads>>>(_atoms.d_y,d_y,N);
    copy_ker<<<blocks,threads>>>(_atoms.d_vx,d_vx,N);
    copy_ker<<<blocks,threads>>>(_atoms.d_vy,d_vy,N);
    copy_ker<<<blocks,threads>>>(_atoms.random_fx,random_fx,N);
    copy_ker<<<blocks,threads>>>(_atoms.random_fy,random_fy,N);
  }

  ~Atoms(){
    //free host memory
    free(x);free(y);free(vx);free(vy);
    //free device memory
    cudaFree(d_x);cudaFree(d_y);cudaFree(d_vx);cudaFree(d_vy);cudaFree(random_fx);cudaFree(random_fy);
  }
};

/*__global__ void Velocity_conf_zero(Atoms atom,int N){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if(idx < N){
    atom.d_vx[idx] = 0.0;
    atom.d_vy[idx] = 0.0;
    atom.d_x[idx] = 0.0;
    atom.d_y[idx] = 0.0;
  }
}

__global__ void Initial_conf_kernel(Atoms atom,double L,int N,curandState *random_x,curandState *random_y){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if(idx < N){
    atom.d_x[idx] = L * curand_uniform_double(&random_x[idx]);
    atom.d_y[idx] = L * curand_uniform_double(&random_y[idx]);
  }
}

void Initial_conf(Atoms atom,double L,int N){
  int threads = 1024;
  int blocks = (N -1+threads)/threads;
  random_device seed;
  curandState *random_x,*random_y;
  cudaMalloc(&random_x,N * sizeof(curandState));
  cudaMalloc(&random_y,N * sizeof(curandState));
  setCurand<<<blocks,threads>>>(seed(), random_x);
  setCurand<<<blocks,threads>>>(seed(), random_y);

  Initial_conf_kernel<<<blocks,threads>>>(atom,L,N,random_x,random_y);

  cudaFree(random_x);cudaFree(random_y);
}*/
