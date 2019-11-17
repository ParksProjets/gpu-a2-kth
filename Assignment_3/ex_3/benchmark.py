#!/usr/bin/env python3

# On PDC, load Python3 using 'load anaconda/py36/5.0.1'.

import subprocess as sp
import csv
import re
import sys


# Benchmark data.
SIZES = list(range(64, 4097, 64))
VERSIONS = ["CPU matmul", "global memory", "shared memory", "cuBLAS"]


def find_time(text, helper):
    "Find execution time for given helper."

    regex = "^.*%s.*:\s*([0-9.]+) ms" % helper
    match = re.search(regex, text, re.M)
    return float(match.group(1))


def run_ps(size):
    "Run the process and return the time."

    ps = sp.Popen(["./exercise_3", "-s", str(size), "-v"], stdout=sp.PIPE)
    out, _ = ps.communicate()

    text = out.decode()
    return [find_time(text, helper) for helper in VERSIONS]


def main():
    "Entry point of this program."

    with open("results.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Input size", "CPU (ms)", "GPU Global (ms)",
            "GPU Shared (ms)", "GPU cuBLAS (ms)"])

        for size in SIZES:
            results = run_ps(size)
            writer.writerow([size, *results])


if __name__ == "__main__":
    main()
