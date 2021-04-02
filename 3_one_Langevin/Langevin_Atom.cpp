#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <fstream>
#include <curand.h>
#include <curand_kernel.h>
#include "particle.cuh"

using namespace std;

//粒子数
#define N 1

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

//Ligevin eq
__global__ void eq_motion(Atoms atom,double dt,double mass){
  uint idx = threadIdx.x + blockIdx.x*blockDim.x;
  atom.d_vx[idx] += -atom.d_vx[idx] * dt /mass + sqrt(2. *dt)* curand_normal_double(&atom.random_fx[idx])/mass;
  atom.d_vy[idx] += -atom.d_vy[idx] * dt /mass + sqrt(2. *dt)* curand_normal_double(&atom.random_fy[idx])/mass;
  atom.d_x[idx] += atom.d_vx[idx]*dt;
  atom.d_y[idx] += atom.d_vy[idx]*dt;
}

void E_15_ofstream(ofstream *file){
  char filename[256];
  sprintf(filename,"time_config_speed_atom.dat");

  (*file).open(filename);
  vector<ofstream*> of_file{file};

  for(int i = 0;i < 1;i++){
    (*of_file[i]).setf(ios::scientific);
    (*of_file[i]).precision(15);
  }
}

void output(Atoms atom,ofstream *file,double t){
  int size = N * sizeof(double);
  copyD2H(atom.x,atom.d_x,size);
  copyD2H(atom.y,atom.d_y,size);
  copyD2H(atom.vx,atom.d_vx,size);
  copyD2H(atom.vy,atom.d_vy,size);
  for(int k = 0;k < N;k++){
    *file << t << " "<< atom.x[k] << " " << atom.y[k] << " " << atom.vx[k] << " " << atom.vy[k] << endl;
  }
}

int main(){
  double t;
  double dt = 0.01,mass = 1.,time_max = 100.;
  int threads = 1;
  int blocks = 1;

  Atoms atom(N);
  atom.Velocity_conf_zero();

  ofstream file;
  E_15_ofstream(&file);
  for(t = 0;t <= time_max;t += dt){
    eq_motion<<<blocks,threads>>>(atom,dt,mass);
    output(atom,&file,t);
  }

  file.close();
}
