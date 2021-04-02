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

__global__ void device_add(Myvec &v){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if(idx < N){
    v.d_c[idx] = v.d_a[idx] + v.d_b[idx];
  }
}

__global__ void fillarr_device(Myvec &v){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if(idx < N){
    v.d_x[idx] = 1;
    v.d_y[idx] = 2;
  }
}

void print_equation(int *a,int *b,int *c){
  for(int i = 0;i < N;i++){
    printf("%d + %d = %d\n",a[i],b[i],c[i]);
  }
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

struct Myvec{
  int N;
  double *x,*y,*z;
  double *d_x,*d_y,*d_z;

  Myvec(int _N){
    N = _N;
    int size = N * sizeof(int);

    //シード値
    /*random_device seed_gen;

    int threads = 1;
    int blocks = 1;*/

    //allocate host memory
    x = (int*)malloc(size);
    y = (int*)malloc(size);
    z = (int*)malloc(size);

    //allocate device memory
    cudaMalloc(&d_x,size);
    cudaMalloc(&d_y,size);
    cudaMalloc(&d_z,size);

  }

  //コピーコンストラクタ
  Myvec(const Myvec &_vec){
    int threads = 1024;
    int blocks = (N-1)/threads + 1;

    int size = N * sizeof(int);

    N = _atoms.N;
    //allocate host memory
    x = (int*)malloc(size);
    y = (int*)malloc(size);
    z = (int*)malloc(size);

    //allocate device memory
    cudaMalloc(&d_x,size);
    cudaMalloc(&d_y,size);
    cudaMalloc(&d_z,size);

    copy(_vec.x,x,N);
    copy(_vec.y,y,N);
    copy(_vec.z,z,N);
    copy_ker<<<blocks,threads>>>(_vec.d_x,d_x,N);
    copy_ker<<<blocks,threads>>>(_vec.d_y,d_y,N);
    copy_ker<<<blocks,threads>>>(_vec.d_z,d_z,N);

  }

  ~Myvec(){
    //free host memory
    free(x);free(y);free(z);
    //free device memory
    cudaFree(d_x);cudaFree(d_y);cudaFree(d_z);
  }
};

int main(){
  Timer timer;
  Myvec vec(N);

  timer.start_record();

  int threads = 1024;
  int blocks = (N-1)/threads + 1;

  fillarr_device<<<blocks,threads>>>(vec);
  device_add<<<blocks,threads>>>(vec);

  copyD2H(vec.c,vec.d_c,size);
  copyD2H(vec.a,vec.d_a,size);
  copyD2H(vec.b,vec.d_b,size);

  timer.stop_record();

  print_equation(vec.a,vec.b,vec.c);

  timer.print_result();

}
