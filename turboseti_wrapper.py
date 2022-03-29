#!/usr/bin/env python
"""
A wrapper that just runs the inner part of turboSETI that does the Taylor tree stuff.
"""
import cupy as cp

import turbo_seti
from turbo_seti.find_doppler.kernels import Kernels

assert __name__ == "__main__"

kernels = Kernels(gpu_backend=True, precision=1)
assert kernels.gpu_backend


HEIGHT = 4
WIDTH = 16


def value(row, col):
    if col - row == 7:
        return 1
    return 0


arr = cp.array(
    [value(x // WIDTH, x % WIDTH) for x in range(HEIGHT * WIDTH)], dtype=float
)
bin_height = f"{HEIGHT:b}"
assert bin_height.count("1") == 1


def row_index_bit_reverse(n):
    assert 0 <= n < HEIGHT
    n_bits = f"{(HEIGHT + n):b}"[1:]
    assert len(n_bits) == bin_height.count("0")
    return int("".join(reversed(n_bits)), 2)


def unshuffle_array(arr):
    """
    Turboseti stores its tree in such a way that the row is indexed by the bit-reversed drift.
    This unshuffling converts the turboseti output to an array indexed by
    arr[drift rate][start column]
    """
    shuffled = arr.reshape((HEIGHT, WIDTH))
    output = cp.empty((HEIGHT, WIDTH))
    for i in range(HEIGHT):
        j = row_index_bit_reverse(i)
        output[i] = shuffled[j]
    return output


print("array:")
print(arr.reshape((HEIGHT, WIDTH)))
kernels.tt.flt(arr, HEIGHT * WIDTH, HEIGHT)
output = unshuffle_array(arr)
print("output:")
print(output)
