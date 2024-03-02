## Use grid-stride loop to manipulate large or arbitrary-size array

A classic use of CUDA kernel is to parallelizing a loop. A premise to the efficient parallelization is to launch enough threads to fully utilize the GPU. Typically, we want to launch one thread per data element. As an example pattern, here is a typical monolithic kernel:

```
___global__ myKernel(int N, float* x, float* y)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;       
    if (idx < N)
        y[i] = x[i] + y[i]

}

```
A monolithic kernel uses a single large grid of threads to process the entire array in one pass. This kernel works only if your data elements is not more than the threads available. But if that cannot be assured or taken for granted during the run time, then you need to implement this in a loop. In such loop, each index will skip to process next element, until all the data elements are processed. This implementation is known as grid-stride. 

So assume you have N data element, and P threads, and N is much bigger than P. So as far as N data elements concerns, the first P elements will be processed by P threads. Once these first P elements are done, next P elements will be processed by P threads, and so on. This continues until all N data elements are processed. In such scenario, the first thread will skip to next element that's P elements away. Therefore, how many element to skip corrcorresponds to stride. 

It turns out the way to calculate the stride is very simple: 

`threads/block * block/grid = threads/grid`

where `threads/grid` is how much threads you have to work with.

Using available index that reflects the formula above:
// do something
```
blockDim.x * gridDim.x = stride
```

So `myKernel` above is now implemented with loop stride:

```
___global__ myKernel(int N, float* x, float* y)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;    
    int stride = blockDim.x * gridDim.x;  
    for (int i = idx, i < N, i += stride)
    {
            y[i] = x[i] + y[i]
    }

}
```

In every iteration by this `for` loop, all threads in this grids are used to process consecutive elements. Then in the next iteration, i is incremented by `stride`, and this thread then process the next element that's `stride` elements away from the current one.

## Reference 

[Grid stride loop](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/)