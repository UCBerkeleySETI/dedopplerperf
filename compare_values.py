#!/usr/bin/env python
import math


def get_values(filename):
    for line in open(filename):
        for s in line.strip().split():
            yield float(s)


assert __name__ == "__main__"

c_values = list(get_values("./c_values.txt"))
py_values = list(get_values("./py_values.txt"))

assert len(c_values) == len(py_values)

relative_diffs = [abs((c - p) / p) for c, p in zip(c_values, py_values) if p]

print("average disagreement:", sum(relative_diffs) / len(relative_diffs))
