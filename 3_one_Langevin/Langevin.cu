#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <fstream>
#include <curand.h>
#include <curand_kernel.h>

using namespace std;

//粒子数
#define N 1

void copyH2D(void* dest,void* src,std::size_t size){
  cudaMemcpy(dest,src,size,cudaMemcpyHostToDevice);
}

void copyD2H(void* dest,void* src,std::size_t size){
  cudaMemcpy(dest,src,size,cudaMemcpyDeviceToHost);
}

__global__ void Init(double *d_x,double *d_y,double *d_vx,double *d_vy){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if(idx < N){
    d_x[idx] = 0.0;
    d_y[idx] = 0.0;
    d_vx[idx] = 0.0;
    d_vy[idx] = 0.0;
  }
}

//curandStateの初期化
__global__ void setCurand(unsigned long long seed, curandState *state){
    uint i_global = threadIdx.x + blockIdx.x*blockDim.x;
    curand_init(seed, i_global, 0, &state[i_global]);
}

//Ligevin eq
__global__ void eq_motion(double *d_x,double *d_y,double *d_vx,double *d_vy,double dt,double mass,curandState *state_x,curandState *state_y){
  uint idx = threadIdx.x + blockIdx.x*blockDim.x;
  d_vx[idx] += -d_vx[idx] * dt /mass + sqrt(2. *dt)* curand_normal_double(&state_x[idx])/mass;
  d_vy[idx] += -d_vy[idx] * dt /mass + sqrt(2. *dt)* curand_normal_double(&state_y[idx])/mass;
  d_x[idx] += d_vx[idx]*dt;
  d_y[idx] += d_vy[idx]*dt;
}

void output(double *x,double *y,double *vx,double *vy,double t){
  ofstream file;
  char filename[256];
  sprintf(filename,"time_config_speed.dat");
  (file).open(filename);
  (file).setf(ios::scientific);
  (file).precision(15);
  for(int k = 0;k < N;k++){
    file << t << " "<< x[k] << " " << y[k] << " " << vx[k] << " " << vy[k] << endl;
  }
}

int main(){
  double t;
  double dt = 0.01,mass = 1.,time_max = 100.;
  int threads = 1;
  int blocks = 1;
  //シード値
  random_device seed_gen;
  int size = N*sizeof(double);

  //乱数の状態
  curandState *state_x,*state_y;

  //速度、位置
  double *x,*y,*vx,*vy;
  double *d_x,*d_y,*d_vx,*d_vy;

  cudaMalloc(&state_x,N * sizeof(curandState));
  cudaMalloc(&state_y,N * sizeof(curandState));

  //allocate host memory
  x = (int*)malloc(size);
  y = (int*)malloc(size);
  vx = (int*)malloc(size);
  vy= (int*)malloc(size);

  //allocate device memory
  cudaMalloc(&d_x,size);
  cudaMalloc(&d_y,size);
  cudaMalloc(&d_vx,size);
  cudaMalloc(&d_vy,size);

  //シードを使ってcurandStateを初期化
  setCurand<<<blocks,threads>>>(seed_gen(), state_x);
  //シードを使ってcurandStateを初期化
  setCurand<<<blocks,threads>>>(seed_gen(), state_y);

  //初期化
  Init<<<blocks,threads>>>(d_x,d_y,d_vx,d_vy);

  for(t = 0;t <= time_max;t += dt){
    eq_motion<<<blocks,threads>>>(d_x,d_y,d_vx,d_vydt,mass,state_x,state_y);
    copyD2H(x,d_x,size);
    copyD2H(y,d_y,size);
    copyD2H(vx,d_vx,size);
    copyD2H(vy,d_vy,size);
    output(x,y,vx,vy,t);
  }

  //free host memory
  free(x);free(y);free(vx);free(vy);
  //free device memory
  cudaFree(d_x);cudaFree(d_y);cudaFree(d_vx);cudaFree(d_vy);cudaFree(state_x);cudaFree(state_y);
}
