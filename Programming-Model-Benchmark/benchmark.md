## Benchmark a kernel

This example demonstrates how to benchmark a CUDA kernel for multiple runs or iterations. This is done by wrapping a kernel execution in a loop.Once the loop is completed, it shows how to generate statistics such as latency and throughput. The source code for this example is [here](benchmark.cu)

## Instruction

To run the source code, go to `Programming-Model-Benchmark` directory, first compile the source code into an executable:

```
nvcc -o benchmark benchmark.cu
```

Then in the same directory, run the executable:

```
./benchmark.cu
```

and expect output such as this:

```
Average latency: 0.0068048 ms
Time taken by the kernel: 0.006144 ms
p95 latency: 0.007168 ms
p99 latency: 0.019456 ms
Throughput: 146955 ops/s
```