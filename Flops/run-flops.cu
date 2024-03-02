#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>

#define numPtcls (512*128) //Total number of particles
#define threadsPerBlock (128) //Number of threads per block
#define BLOCKS numPtcls / threadsPerBlock//total number of blocks
#define niters (10000) // FMAD iterations per thread


struct Particles {
    float testVariable;
};

__global__ void cudaFunction(Particles *particle)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 1;

    float position = particle[0].testVariable;

    for (int i = 0; i < niters; i++) {
        sum *= position;
    }

    particle[0].testVariable = sum;
}

int main()
{
    Particles *particle = new Particles[numPtcls];

    particle[0].testVariable = 1;

    Particles *device_location;//POINTER TO MEMORY FOR CUDA
    int size = numPtcls * sizeof(Particles);//SIZE OF PARTICLE DATA TO MAKE ROOM FOR IN CUDA
    cudaMalloc((void**)&device_location, size);// allocate device copies
    cudaMemcpy(device_location, particle, size, cudaMemcpyHostToDevice);// copy inputs to device

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float flopcount = float(niters) * float(numPtcls);

    for(int i=0; i<10; i++) {
        cudaEventRecord(start, 0);
        cudaFunction << <BLOCKS, threadsPerBlock >> > (device_location);//CUDA CALL
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaMemcpy(particle, device_location, size, cudaMemcpyDeviceToHost);

    float gpu_time_used;
    cudaEventElapsedTime(&gpu_time_used, start, stop);
        std::cout << std::fixed << std::setprecision(6) << 1e-6 * (flopcount / gpu_time_used) << std::endl;
    }

    cudaFree(device_location);//FREE DEVICE MEMORY
    delete[] particle;//FREE CPU MEMORY

    return 0;
}