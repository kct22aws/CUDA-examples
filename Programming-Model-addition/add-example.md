## [Create a function for GPU to add two vectors](#intro)
This example shows how to create a function that runs as a CUDA kernel in a GPU. There are several important point in this example. The source code for this exercise is [here](./add-example.cu).

## Explanation

1. Memory arrangement in device

`cudaMalloc` function allocates memory on GPU for the tensor of interest. This function takes a pointer and allocate designated device memory per that pointer.
This `cudaMalloc` function is applied to all the tensors involved in this example, including the output tensor. In CUDA programming, typically, memory for output tensors have to be allocated ahead of time, i.e., here, we allocate memory for `dev_c` before the function call to `add` takes place.

`cudaMemcpy` function copies content of memory between a source and target. 



## Instruction

To run the source code, go to `kernel` directory, first compile the source code into an executable:

```
nvcc -o add-example add-example.cu
```

Then in the same directory, run the executable:

```
./add-example
```

and expect output such as this:



