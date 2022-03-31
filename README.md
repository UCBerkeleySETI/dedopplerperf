# dedopplerperf
Performance testing of different dedoppler kernels. There are currently three different dedoppler
kernels tested here, the CudaTaylor5demo code in C+cuda, the turboSETI Python code, and `raw_kernel.py` which is Python + cuda using the cupy RawKernel interface.

They are set up to run on the same input data. Check the comments in the code to see how to
alter it.

I used Ubuntu for testing. Performance on a 256 x 2^19 matrix with a
GTX 1080:

* cudataylor5demo: 0.04s
* raw_kernel: 0.09s
* turboseti: 0.45s

# How to run CudaTaylor5demo

You need to install the Cuda toolkit, make sure `nvcc` is on your
path, then:

```
nvcc CudaTaylor5demo.cu
./a.out
```

# How to run the Python dedoppler code

You need a Python environment with cupy and turboseti installed. I
provided an `environment.yml` if you are using conda. Then:

```
./turboseti_wrapper.py
./raw_kernel.py
```
