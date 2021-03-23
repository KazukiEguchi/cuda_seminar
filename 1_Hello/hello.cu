#include <stdio.h>
#include <stdlib.h>

__global__ void print_from_device(void){
	printf('Hello World! from device\n');
}

int main(){
	printf("Hello World! from host\n");

	print_from_device<<<1,10>>>();
	cudaDeviceSynchronize();
	return 0;
}
