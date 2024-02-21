## [Handle data and thread mismatch](#intro)
Due to GPU hardware design, it is ideal to have a block of thread at multiple of 32 (ie, 256) for performance benefit. Let's say you want threads per block to be 256, and you have N = 1000,000 data elements. But 1000,000 is not completely divisible by 32. So you cannot allocate exact number of blocks to handle these 1000,000 data elements. In any case, you will have a block whose threads are not completely used. 

To calculate the number of blocks needed without wasting more blocks than necessary, this is how to do it:


```
size_t number_of_blocks = (N + threads_per_block -1) / nuthreads_per_block
 ```

 Why do we need to subtract 1? As it turns out, it only matters if N is completely divisible by number_per_block. If you don't subtract 1, you will end up with one extra block that is not used. 

 Try N = 1000,000, threads_per_block = 32, and you will find out that if you don't subtract 1, you will have 31251 blocks. With 32 threads per block, you will end up allocating 1000,032 threads. But if you subtract 1 as shown in the formula above, you will end up with 31250 blocks, multiply it by 32 threads per block, you will allocate exactly 1000,000 threads.

 Here is a simple test program:

 ```
#include <stdio.h>

int main() {
    // Assume `N` is known
    int N = 1000000;
    // int *a;
    // size_t size = N * sizeof(int);
    
    // cudaMallocManaged(&a, size);
    
    // Assume we have a desire to set `threads_per_block` exactly to `256`
    size_t threads_per_block = 32;

    // Ensure there are at least `N` threads in the grid, but only 1 block's worth extra
    size_t number_of_blocks = (N + threads_per_block -1 ) / threads_per_block;

    printf("Number of blocks %zu \n", number_of_blocks);
    
    return 0;
}
 
 ```