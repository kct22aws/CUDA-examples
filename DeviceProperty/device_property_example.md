## [Device coniguration example use](#intro)
Now that we know how to retrieve GPU configurations, what can we do with it? One of the examples is when you are writing an application taht depends on certain
requirements in these configurations. Suppose you want to write an application to carry out Bfloat16 precision floating-point operations. For this requirement, 
you need to make sure the GPU meets the compute capability requirement. From [the table](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=compute%20capability#features-and-technical-specifications) in CUDA documentation, you see that it requires compute capability of 8 or greater.

The source code is [here](./search-config.cu).

## Explanation

In the example code, `cudaDeviceProp` object is used to hold the GPU (device) configuration. The object `prop` is then initialized with standard C library `memset` function.
Then `cudaChooseDevice` will find the device that matches the requirement, which is specified by the major and minor versions of compute capability. Finally, `cudaSetDevice` will set the device that matches the requirement for all active host threads. Therefore the application to run Bfloat16 operations will be assured to run on this particular deivce.

## Instruction

In the same directory as the script [here](./search-config.cu), first generate an executable by compiling the program:

```
nvcc -o search-config search-config.cu
```
This will generate `search-config` excutable. Now run this executable:

```
./search-config
```
and expect output like:
```
Current Device ID: 0
CUDA DEVICE ID that best matches the requirement: 0
```

This is example is not often used nowadays, as it is very rare that you would have a GPU system with different GPU models. With the prevalence of GPU in public cloud vendors, you will find GPU instances to have same GPU models clustered together in one server. 