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

__global__ void Init_conf(double *d_x,double *d_y,double L,curandState *random_x,curandState *random_y){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if(idx < N){
    d_x[idx] = L * curand_uniform_double(&random_x[idx]);
    d_y[idx] = L * curand_uniform_double(&random_y[idx]);
  }
}

__global__ void Init_vel(double *d_vx,double *d_vy){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if(idx < N){
    d_vx[idx] = 0.0;
    d_vy[idx] = 0.0;
  }
}

__global__ void Init_sigma(double *d_r){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if(idx < N / 2) d_r[idx] = 1.0;
  else if(idx < N) d_r[idx] = 1.4;
}

void Init(double *d_x,double *d_y,double *d_vx,double *d_vy,double *d_r,double L){
  random_device seed;
  curandState *random_x,*random_y;
  cudaMalloc(&random_x,N * sizeof(curandState));
  cudaMalloc(&random_y,N * sizeof(curandState));
  setCurand<<<blocks,threads>>>(seed(), random_x);
  setCurand<<<blocks,threads>>>(seed(), random_y);

  Init_conf<<<blocks,threads>>>(d_x,d_y,L,random_x,random_y);
  Init_vel<<<blocks,threads>>>(d_vx,d_vy);
  Init_sigma<<<blocks,threads>>>(d_r);

  cudaFree(random_x);cudaFree(random_y);
}

void E_15_ofstream(ofstream *file){
  char filename[256];
  sprintf(filename,"test_config_speed.dat");

  (*file).open(filename);
  vector<ofstream*> of_file{file};

  for(int i = 0;i < 1;i++){
    (*of_file[i]).setf(ios::scientific);
    (*of_file[i]).precision(15);
  }
}

void output_conf(double *x,double *y,double *vx,double *vy,double *r,double *d_x,double *d_y,double *d_vx,double *d_vy,double *d_r,int size,ofstream *file){
  copyD2H(x,d_x,size);
  copyD2H(y,d_y,size);
  copyD2H(vx,d_vx,size);
  copyD2H(vy,d_vy,size);
  copyD2H(r,d_r,size);
  for(int k = 0;k < N;k++){
    *file << x[k] << " " << y[k] << " " << vx[k] << " " << vy[k] << endl;
  }
}

int main(){
  //シード値
  random_device seed_gen;
  int size = N*sizeof(double);
  double density = 0.8;
  double L = sqrt(double(N)/density);

  //乱数の状態
  curandState *random_fx,*random_fy;

  //速度、位置
  double *x,*y,*vx,*vy,*r;
  double *d_x,*d_y,*d_vx,*d_vy,*d_r;

  cudaMalloc(&random_fx,N * sizeof(curandState));
  cudaMalloc(&random_fy,N * sizeof(curandState));

  //allocate host memory
  x = (double*)malloc(size);
  y = (double*)malloc(size);
  vx = (double*)malloc(size);
  vy= (double*)malloc(size);
  r = (double*)malloc(size);

  //allocate device memory
  cudaMalloc(&d_x,size);
  cudaMalloc(&d_y,size);
  cudaMalloc(&d_vx,size);
  cudaMalloc(&d_vy,size);
  cudaMalloc(&d_r,size);

  //シードを使ってcurandStateを初期化
  setCurand<<<blocks,threads>>>(seed_gen(), random_fx);
  //シードを使ってcurandStateを初期化
  setCurand<<<blocks,threads>>>(seed_gen(), random_fy);

  //初期化
  Init(d_x,d_y,d_vx,d_vy,d_r,L);

  ofstream file;
  E_15_ofstream(&file);

  output_conf(x,y,vx,vy,r,d_x,d_y,d_vx,d_vy,d_r,size,&file);

  file.close();
  //free host memory
  free(x);free(y);free(vx);free(vy);free(r);
  //free device memory
  cudaFree(d_x);cudaFree(d_y);cudaFree(d_vx);cudaFree(d_vy);cudaFree(d_r);(random_fx);cudaFree(random_fy);
}
