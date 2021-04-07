#include "para_useful.cuh"

__global__ void genrand_kernel(double *result, curandState *state){
    uint i_global = threadIdx.x + blockIdx.x*blockDim.x;
    result[i_global] = curand_normal_double(&state[i_global]);
}

int main(){
  double *x;
  double *d_x;
  curandState *state;

  double av = 0.0;
  double sigma = 0.0;

  x = (double*)malloc(N * sizeof(double));
  cudaMalloc(&d_x, N * sizeof(double));
  cudaMalloc(&state, N * sizeof(curandState));

  cuda_random_set(state);

  genrand_kernel<<<blocks,threads>>>(d_x,state);
  copyD2H(x,d_x,N*sizeof(double));

  for(int i = 0;i < N;i++){
    av += x[i];
  }
  av /= N;
  for(int i = 0;i < N;i++){
    sigma += (x[i]-av)*(x[i]-av);
  }
  sigma /= N;
  sigma = sqrt(sigma);

  cout << av << " " << sigma << endl;

  free(x);
  cudaFree(d_x);
  cudaFree(state);
}
