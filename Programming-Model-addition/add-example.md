## [Create a function for GPU to add two vectors](#intro)
This example shows how to create a function that runs as a CUDA kernel in a GPU. There are several important point in this example. The source code for this exercise is [here](./gpu-add.cu).

## Explanation

### Memory arrangement in device

`cudaMalloc` function allocates memory on GPU for the tensor of interest. This function takes a pointer and allocate designated device memory per that pointer.
This `cudaMalloc` function is applied to all the tensors involved in this example, including the output tensor. In CUDA programming, typically, memory for output tensors have to be allocated ahead of time, i.e., here, we allocate memory for `dev_c` before the function call to `add` takes place.

`cudaMemcpy` function copies content of memory between a source and target. The input arguments follows this order: destination, source, and kind of transfer. In this case, we want to copy `a` into
`dev_a`, by allocating number of bytes of the data type; the direction of copy is designated by `cudaMemcpyHostToDevice`, which indicates we want to copy array `a` from host to device.

`cudaFree` function returns the allocated memory in device back to a pool of available memory.

### Kernel function taking on pointers as inputs

Pointers are used as input arguments to kernel. Why do we pass by pointer to the kernel and not pass by value? There are these reasons:

In CUDA, when writing a kernel, it is common to pass parameters by pointer rather than by value for performance reasons. Here are a few reasons for this:

1. **Memory Efficiency:** Passing parameters by pointer allows kernels to directly access data in the global memory of the GPU. This avoids the need to make additional copies of data, which can be time-consuming and memory-intensive.

2. **Avoiding Data Transfer Overhead:** When you pass parameters by value, the values need to be transferred from the host (CPU) to the device (GPU). This involves copying data over the PCI Express bus, which can introduce significant overhead. By passing a pointer, you can work directly with the data already present in the GPU memory.

3. **Shared Memory Access:** Kernels in CUDA often utilize shared memory, which is a fast and low-latency memory space that is shared among threads in a thread block. Passing by pointer allows threads within a block to efficiently share data through shared memory.

4. **Mutable Data:** If you need to modify the values of parameters within the kernel and have those modifications reflected in the calling function, passing by pointer is necessary. Pass by value would only modify a local copy of the parameter within the kernel.

5. **Array Processing:** GPUs are designed for parallelism, and passing pointers allows easy access to arrays and enables parallel processing of array elements.

While passing by pointer offers performance benefits, it's essential to manage memory access carefully to avoid data hazards and ensure proper synchronization among threads. Additionally, developers need to consider issues like coalesced memory access patterns for optimal performance.

## Instruction

To run the source code, go to `Programming-Model-addition` directory, first compile the source code into an executable:

```
nvcc -o gpu-add gpu-add.cu
```

Then in the same directory, run the executable:

```
./gpu-add
```

and expect output such as this:

```
0 + 0 = 0
-1 + 1 = 0
-2 + 4 = 2
-3 + 9 = 6
-4 + 16 = 12
-5 + 25 = 20
-6 + 36 = 30
-7 + 49 = 42
-8 + 64 = 56
-9 + 81 = 72
``````
