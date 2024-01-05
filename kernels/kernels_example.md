## [Run a function in GPU](#intro)
In CUDA programming, it is necessary to specify which part of your code runs in CPU (host), and which parts runs in 
GPU (device). Therefore it is a good idea to encapsulate your code for GPU in a function. This function is also known as a "kernel". In a typical CUDA program, host code makes a kernel call to the function, and this function will be executed in the device. Once it's done, the results will be copied from the device to the host.

## Explanation
In order for a function to execute on the device, it needs to be labeled with a qualifier `__global__`:

```
__global__ void gpufunc () {
    .....
}
```

and in the host code, where such function is invoked:

```
gpufunc<<<p, q>>>()
```
Such function or kernell call needs to have `<<<p, q>>>` in the call, where

p is number of blocks\
q is number of threads

Blocks are the basic units that execute in parallel in a GPU program.

The source code for this exercise is [here](./kernel-demo.cu).

## Instruction

To run the source code, go to `kernel` directory:

```
nvcc -o kernel-demo kernel-demo.cu
```

Then in the same directory, run the executable:

```
./kernel-demo
```

and expect output such as this:

```
...
-9997 + 99940009 = 99930012
-9998 + 99960004 = 99950006
-9999 + 99980001 = 99970002
```


