#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <fstream>
#include <curand.h>
#include <curand_kernel.h>
#include "particle.cuh"

using namespace std;

//粒子数
const int N =1;

void copyH2D(void* dest,void* src,std::size_t size){
  cudaMemcpy(dest,src,size,cudaMemcpyHostToDevice);
}

void copyD2H(void* dest,void* src,std::size_t size){
  cudaMemcpy(dest,src,size,cudaMemcpyDeviceToHost);
}

//Ligevin eq
__global__ void eq_motion(Atoms &atom,double dt,double mass){
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

void output(Atoms &atom,ofstream *file,double t){
  int size = N * sizeof(double);
  copyD2H(atom.x,atom.d_x,size);
  copyD2H(atom.y,atom.d_y,size);
  copyD2H(atom.vx,atom.d_vx,size);
  copyD2H(atom.vy,atom.d_vy,size);
  for(int k = 0;k < N;k++){
    *file << t << " "<< atom.x[k] << " " << atom.y[k] << " " << atom.vx[k] << " " << atom.vy[k] << endl;
  }
}

__global__ void Velocity_conf_zero(Atoms &a,int N){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if(idx < N){
    a.d_x[idx] = 1.0;
    a.d_y[idx] = 1.0;
    a.d_vx[idx] = 1.0;
    a.d_vy[idx] = 1.0;
  }
}

int main(){
  double t;
  double dt = 0.01,mass = 1.,time_max = 100.;
  int threads = 1;
  int blocks = 1;

  Atoms atom(N);
  Velocity_conf_zero<<<blocks,threads>>>(atom,N);

  ofstream file;
  E_15_ofstream(&file);
  for(t = 0;t <= time_max;t += dt){
    eq_motion<<<blocks,threads>>>(atom,dt,mass);
    output(atom,&file,t);
  }

  file.close();
}
