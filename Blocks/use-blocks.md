## Set up CUDA blocks
In CUDA programming, a GPU kernel is executed by a grid of thread blocks. Each thread block contains multiple threads, and these threads collaborate to perform computations in parallel. The relationship between blocks, threads, and memory in CUDA is crucial for efficient parallel processing.

1. **Thread Blocks:**
   - A grid is made up of thread blocks.
   - Thread blocks are organized in a grid structure and are scheduled for execution on a streaming multiprocessor (SM) of the GPU.

2. **Threads:**
   - Each thread block contains a set of threads.
   - Threads within a block can communicate and synchronize with each other using shared memory.

3. **Memory Hierarchy:**
   - Each thread has its own private local memory.
   - Threads within a block can share data through fast, but limited, shared memory.
   - Global memory is accessible by all threads and persists throughout the kernel's execution.

4. **Indexing:**
   - Threads are identified by their global and block indices.
   - These indices are used to map computations to specific threads and blocks.

Understanding these relationships is essential for optimizing memory access patterns and ensuring efficient parallelization of tasks in CUDA programs. It allows developers to design algorithms that effectively leverage the parallel architecture of the GPU.

Thread index is defined in a kernel as (1-D example):

```
idx = blockIdx.x * blockDim.x + threadIdx.x
```

Where 

`lockIdx.x` is the index of this block, range starts at 0.
`blockDim.x` is how many threads there are per block.
`threadIdx.x`is the ID of a thread.

Therefore. `idx` is the value that uniquely identifies a thread, given the finite number of blocks designated in this CUDA kernel.


