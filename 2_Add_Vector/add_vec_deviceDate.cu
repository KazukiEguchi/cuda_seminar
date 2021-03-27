#include <stdio.h>
#include <stdlib.h>
#include "timer.cuh"

//ベクトルの要素数
#define N 1000000

void copyH2D(void* dest,void* src,std::size_t size){
  cudaMemcpy(dest,src,size,cudaMemcpyHostToDevice);
}

void copyD2H(void* dest,void* src,std::size_t size){
  cudaMemcpy(dest,src,size,cudaMemcpyDeviceToHost);
}

__global__ void device_add(int *a,int *b,int *c){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if(idx < N){
    c[idx] = a[idx] + b[idx];
  }
}

__global__ fillarr_device(int *a,int *b){
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
  int *d_a,*d_b,*d_c;
  int size = N*sizeof(int);

  timer.start_record();

  //allocate host memory
  a = (int*)malloc(size);
  b = (int*)malloc(size);
  c = (int*)malloc(size);

  //allocate device memory
  cudaMalloc(&d_a,size);
  cudaMalloc(&d_b,size);
  cudaMalloc(&d_c,size);

  copyH2D(d_a,a,size);
  copyH2D(d_b,b,size);

  int threads = 1024;
  int blocks = (N-1)/threads + 1;

  fillarr_device<<<blocks,threads>>>(d_a,d_b);
  device_add<<<blocks,threads>>>(d_a,d_b,d_c);

  copyD2H(c,d_c,size);

  timer.stop_record();

  print_equation(a,b,c);

  timer.print_result();

  //free host memory
  free(a);free(b);free(c);
  //free device memory
  cudaFree(d_a);cudaFree(d_b);cudaFree(d_c);
}
