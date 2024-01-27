# include <stdio.h>
# include <iostream> 
# include "../common/book.h"
# include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <numeric>

#define N   4096

using namespace std;

std::vector<float> executionTimes;

__global__ void add( int *a, int *b, int *c ) {
    int tid = blockIdx.x;    // this thread handles the data at its thread id
    if (tid < N)
        c[tid] = a[tid] + b[tid];
}

int main( void ) {
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;

    
    // allocate the memory on the GPU
    HANDLE_ERROR( cudaMalloc( &dev_a, N * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( &dev_b, N * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( &dev_c, N * sizeof(int) ) );

    // fill the arrays 'a' and 'b' on the CPUcreate a function that runs as
    for (int i=0; i<N; i++) {
        a[i] = -i;
        b[i] = i * i;
    }

    // copy the arrays 'a' and 'b' to the GPU
    HANDLE_ERROR( cudaMemcpy( dev_a, a, N * sizeof(int),
                              cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( dev_b, b, N * sizeof(int),
                              cudaMemcpyHostToDevice ) );

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int numIterations = 100;

    for (int i =0; i<numIterations; ++i) {
        // Record the start event
        cudaEventRecord(start);

        add<<<N,1>>>( dev_a, dev_b, dev_c );

        // Record the stop event
        cudaEventRecord(stop);
        // Synchronize to make sure the events have completed
        cudaEventSynchronize(stop);

        float milliseconds = 0.0f;

        cudaEventElapsedTime(&milliseconds, start, stop);

        executionTimes.push_back(milliseconds);
    }
    // Calculate and print statistics
    std::sort(executionTimes.begin(), executionTimes.end());

    float averageLatency = accumulate(executionTimes.begin(), executionTimes.end(), 0.0) / executionTimes.size();

    std::cout << "Average latency: " << averageLatency << " ms" << std::endl;

    // Calculate the elapsed time
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);


    float p95Latency = executionTimes[static_cast<int>(0.95 * numIterations)];
    float p99Latency = executionTimes[static_cast<int>(0.99 * numIterations)];
    float throughput = 1000 / averageLatency;

    std::cout << "Time taken by the kernel: " << milliseconds << " ms" << std::endl;
    std::cout << "p95 latency: " << p95Latency << " ms" << std::endl;
    std::cout << "p99 latency: " << p99Latency << " ms" << std::endl;
    std::cout << "Throughput: " << throughput << " ops/s" << std::endl;
     
    // copy the array 'c' back from the GPU to the CPU
    HANDLE_ERROR( cudaMemcpy( c, dev_c, N * sizeof(int),
                              cudaMemcpyDeviceToHost ) );

    // display the results
    // for (int i=0; i<N; i++) {
    //     printf( "%d + %d = %d\n", a[i], b[i], c[i] );
    // }

    // free the memory allocated on the GPU
    HANDLE_ERROR( cudaFree( dev_a ) );
    HANDLE_ERROR( cudaFree( dev_b ) );
    HANDLE_ERROR( cudaFree( dev_c ) );

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

