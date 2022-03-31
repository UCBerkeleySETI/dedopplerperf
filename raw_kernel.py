#!/usr/bin/env python
"""
A test using cupy's RawKernel functionality to run cuda code for dedopplering.
"""

import cupy as cp
import time

from performance import *

assert __name__ == "__main__"

taylor_kernel = cp.RawKernel(
    r"""
extern "C" __global__
void taylor(const float* A, float* B, int kmin, int kmax, int set_size, int n_time, int n_freq) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int k = kmin + tid;
  bool worker = (k >= kmin) && (k < kmax) && set_size <= n_time;
  if (!worker) {
    return;
  }
  for (int j = 0; j < n_time; j += set_size) {
    for (int j0 = set_size - 1; j0 >= 0; j0--) {
      int j1 = j0 / 2;
      int j2 = j1 + set_size / 2;
      int j3 = (j0 + 1) / 2;
      if (k + j3 < kmax) {
        B[(j + j0) * n_freq + k] = A[(j + j1) * n_freq + k] + A[(j + j2) * n_freq + k + j3];
      }
    }
  }
}
""",
    "taylor",
)


def taylor_tree(array):
    """
    Runs the Taylor Tree algorithm.
    array should be a 2d array with shape (N_TIME, N_FREQ).
    array starts out with data indexed by [time][frequency].
    It ends up containing sums along lines, indexed by [start time][drift]
    where drift is measured in the number of frequency buckets drifted over
    the entire time of the input data.

    Returns the output of the Taylor tree algorithm. It may or may not be the same array
    as the input. Either way, the input array is overwritten with intermediate values,
    so you can't use it any more.
    """
    assert array.shape == (N_TIME, N_FREQ)
    buf = cp.zeros_like(array)
    start_time = time.time()

    # TODO: we have to set block_size intelligently for actual production.
    # Look through the cupy interface to see how smart we can be about
    # it, find something comparable to cudaOccupancyMaxPotentialBlockSize
    block_size = 1024
    grid_size = (N_FREQ + block_size - 1) // block_size

    set_size = 2
    while set_size <= N_TIME:
        taylor_kernel(
            (grid_size,),
            (block_size,),
            (array, buf, 0, N_FREQ, set_size, N_TIME, N_FREQ),
        )
        array, buf = buf, array
        set_size *= 2

    print(f"time elapsed in doppler loop: {time.time() - start_time:.3f}s")
    return array


arr = make_test_array().reshape((N_TIME, N_FREQ))
print("array:")
show_array(arr)

arr = taylor_tree(arr)

print("output:")
show_array(arr)
