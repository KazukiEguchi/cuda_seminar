#include <stdio.h>
#include <stdlib.h>

//ベクトルの要素数
#define N 512

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

void fillarr(int *a,int num){
  for(int i = 0;i < N;i++){
    a[i] = num;
  }
}

void print_equation(int *a,int *b,int *c){
  for(int i = 0;i < N;i++){
    printf("%d + %d = %d\n",a[i],b[i],c[i]);
  }
}

int main(){
  int *a,*b,*c;
  int *d_a,*d_b,*d_c;
  int size = N*sizeoff(int);

  //allocate host memory
  a = (int*)malloc(size);
  b = (int*)malloc(size);
  c = (int*)malloc(size);

  fillarr(a,1);
  fillarr(b,2);

  //allocate device memory
  cudaMalloc(&d_a,size);
  cudaMalloc(&d_b,size);
  cudaMalloc(&d_c,size);

  copyH2D(d_a,a,size);
  copyH2D(d_b,b,size);

  int blocks = 8;
  int threads = (int)((N-1)/block + 1);

  device_add<<<blocks,threads>>>(d_a,d_b,d_c);

  copyD2H(c,d_c,size);

  print_equation(a,b,c);

  //free host memory
  free(a);free(b);free(c);
  //free device memory
  cudaFree(d_a);cudaFree(d_b);cudaFree(d_c);
}
