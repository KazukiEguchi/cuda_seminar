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
const int NT = 1;
const int NB = 1;

void copyH2D(void* dest,void* src,std::size_t size){
  cudaMemcpy(dest,src,size,cudaMemcpyHostToDevice);
}

void copyD2H(void* dest,void* src,std::size_t size){
  cudaMemcpy(dest,src,size,cudaMemcpyDeviceToHost);
}

//Ligevin eq
__global__ void updateVel_device(double *d_vx,double *d_vy,curandState *fx,curandState *fy,double dt,double mass){
  uint idx = threadIdx.x + blockIdx.x*blockDim.x;
  d_vx[idx] += d_vx[idx] * dt /mass + sqrt(2. *dt)* curand_normal_double(&fx[idx])/mass;
  d_vy[idx] += d_vy[idx] * dt /mass + sqrt(2. *dt)* curand_normal_double(&fy[idx])/mass;
}

void updateVel(Atoms &at,double dt,double mass){
  updateVel_device<<<NB,NT>>>(at.d_vx,at.d_vy,at.random_fx,at.random_fy,dt,mass);
}

__global__ void updatePos_device(double *d_x,double *d_y,double dt){
  uint idx = threadIdx.x + blockIdx.x*blockDim.x;
  d_x[idx] += d_vx[idx]*dt;
  d_y[idx] += d_vy[idx]*dt;
}

void updatePos(Atoms &at,double dt){
  updatePos_device<<<NB,NT>>>(at.d_x,at.d_y,dt);
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

int main(){
  double t;
  double dt = 0.01,mass = 1.,time_max = 100.;

  Atoms atom(N);
  Velocity_conf_zero(atom,N);

  ofstream file;
  E_15_ofstream(&file);
  for(t = 0;t <= time_max;t += dt){
    updateVel(atom,dt,mass);
    updatePos(atom,dt);
    output(atom,&file,t);
  }

  file.close();
}
