#!/usr/bin/env python
"""
A wrapper that just runs the inner part of turboSETI that does the Taylor tree stuff.
"""
import cupy as cp
import math
import time

import turbo_seti
from turbo_seti.find_doppler.kernels import Kernels

assert __name__ == "__main__"

kernels = Kernels(gpu_backend=True, precision=1)
assert kernels.gpu_backend


N_TIME = 256
N_FREQ = 2 ** 19


def get_test_value(time, freq):
    """
    This should match the logic in get_test_value in the C code.
    """
    return ((time * freq) % 1337) * 0.123


arr = cp.array(
    [get_test_value(x // N_FREQ, x % N_FREQ) for x in range(N_TIME * N_FREQ)],
    dtype=float,
)
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


def show_array(arr):
    for row in arr[:16]:
        print("  ".join(f"{x.item():.8f}" for x in row[:8]))


print("array:")
show_array(arr.reshape((N_TIME, N_FREQ)))
start_time = time.time()
kernels.tt.flt(arr, N_TIME * N_FREQ, N_TIME)
print(f"time elapsed in flt: {time.time() - start_time:.3f}s")
output = unshuffle_array(arr)
print("output:")
show_array(output)
