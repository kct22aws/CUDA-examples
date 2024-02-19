## Memory management for kernel
CUDA kernel operates on chunks of data that are placed on GPU. In general, there are these steps to be implemented in `main` function:

1. Determine data type of interest - This refers to the data type of the input data for each kernel.

2. Define a pointer to the input data - Pointer will be used to access the input data for each kernel. For example, if the data type of the input data is `int`, then we may define a pointer:

```
int* aPtr
```

in `main` function prior to calling the kernel. `aPtr` is a pointer that points to an integer.

3. Allocate memory based on size of the input data - Besides the input data type, we also need to know the input data length, i.e., number of elements (`N`) in the input data for a kernel. This typically follows such pattern:

```
size_t size = N * sizeof(int);
cudaMallocManaged(&aPtr, size);
...
/* CALL KERNEL */
...
cudaFree(aPtr)
```

Here we first define `size` as the size of the input object for the kernel. Then pass the address of the pointer `aPtr` with `size` to `cudaMallocManaged` functrion to allocate a chunch of GPU memory. This sets up the memory required for the input data, and tells the kernel where to find input data. After the kernel completes its execution, we have to free the memory poionted by `aPtr` with `cudaFree` function.

