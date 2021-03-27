#include <stdio.h>
#include <stdlib.h>
#include "timer.cuh"

//ベクトルの要素数
#define N 1000000

__global__ void device_add(int *a,int *b,int *c){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if(idx < N){
    c[idx] = a[idx] + b[idx];
  }
}

__global__ void fillarr_device(int *a,int *b){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if(idx < N){
    a[idx] = 1;
    b[idx] = 2;
  }
}

void print_equation(int *a,int *b,int *c){
  for(int i = 0;i < N;i++){
    printf("%d + %d = %d\n",a[i],b[i],c[i]);
  }
}

int main(){
  Timer timer;
  int *a,*b,*c;
  int size = N*sizeof(int);

  timer.start_record();

  //allocate Unified memory
  cudaMallocManaged(&a,size);
  cudaMallocManaged(&b,size);
  cudaMallocManaged(&c,size);

  int threads = 1024;
  int blocks = (N-1)/threads + 1;

  fillarr_device<<<blocks,threads>>>(a,b);
  device_add<<<blocks,threads>>>(a,b,c);
  cudaDeviceSynchronize();

  timer.stop_record();

  print_equation(a,b,c);

  timer.print_result();

  //free device memory
  cudaFree(d_a);cudaFree(d_b);cudaFree(d_c);
}
