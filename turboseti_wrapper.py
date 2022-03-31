#!/usr/bin/env python
"""
A wrapper that just runs the inner part of turboSETI that does the Taylor tree stuff.
"""
import cupy as cp
import math
import time

import turbo_seti
from turbo_seti.find_doppler.kernels import Kernels

from performance import *

assert __name__ == "__main__"

kernels = Kernels(gpu_backend=True, precision=1)
assert kernels.gpu_backend


arr = make_test_array()

bin_height = f"{N_TIME:b}"
assert bin_height.count("1") == 1


def row_index_bit_reverse(n):
    assert 0 <= n < N_TIME
    n_bits = f"{(N_TIME + n):b}"[1:]
    assert len(n_bits) == bin_height.count("0")
    return int("".join(reversed(n_bits)), 2)


def unshuffle_array(arr):
    """
    Turboseti stores its tree in such a way that the row is indexed by the bit-reversed drift.
    This unshuffling converts the turboseti output to an array indexed by
    arr[drift rate][start column]
    """
    shuffled = arr.reshape((N_TIME, N_FREQ))
    output = cp.empty((N_TIME, N_FREQ))
    for i in range(N_TIME):
        j = row_index_bit_reverse(i)
        output[i] = shuffled[j]
    return output


arr = make_test_array()
print("array:")
show_array(arr.reshape((N_TIME, N_FREQ)))
start_time = time.time()
kernels.tt.flt(arr, N_TIME * N_FREQ, N_TIME)
print(f"time elapsed in flt: {time.time() - start_time:.3f}s")
output = unshuffle_array(arr)
print("output:")
show_array(output)
