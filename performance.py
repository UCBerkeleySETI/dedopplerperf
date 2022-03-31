#!/usr/bin/env python
"""
Tools for checking performance of python dedoppler code.
"""
import cupy as cp
import time

N_TIME = 256
N_FREQ = 2 ** 19


def get_test_value(time, freq):
    """
    This should match the logic in get_test_value in the C code.
    """
    return ((time * freq) % 1337) * 0.123


def make_test_array():
    return cp.array(
        [get_test_value(x // N_FREQ, x % N_FREQ) for x in range(N_TIME * N_FREQ)],
        dtype=cp.float32,
    )


def show_array(arr):
    for row in arr[:16]:
        print("  ".join(f"{x.item():.8f}" for x in row[:8]))
