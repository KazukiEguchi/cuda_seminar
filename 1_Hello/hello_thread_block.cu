#include <stdio.h>
#include <stdlib.h>

__global__ void print_from_device(void){
	printf("Hello World! from device (block:%d,threads:%d)\n",blockIdx.x,threadIdx.x);
}

int main(){
  int blocks = 2;
  int threads = 4;
	printf("Hello World! from host\n");

	print_from_device<<<blocks,threads>>>();
	cudaDeviceSynchronize();
	return 0;
}
