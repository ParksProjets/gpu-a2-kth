#!/usr/bin/env python3

# On PDC, load Python3 using 'load anaconda/py36/5.0.1'.

import subprocess as sp
import csv
import re
import sys


# Benchmark data.
NUM_PARTICULES = [
    100, 1000, 5000, 10000, 50000, 100000, 500000,
    1000000, 10000000
]

NUM_ITERATIONS = 1000
BLOCK_SIZES = [16, 32, 64, 128, 256]

# Regex for matching time.
RE_TIME = re.compile(r"^[GC]PU time: ([0-9]+) ms", re.M)


def run(num_particules, block_size, kind):
    "Run the benchmark."

    args = [num_particules, NUM_ITERATIONS, block_size, kind]
    args = [str(v) for v in args]

    ps = sp.Popen(["./exercise_3", *args], stdout=sp.PIPE)

    out, _ = ps.communicate()
    return RE_TIME.search(out.decode()).group(1)


def CPU_benchmark(writer):
    "Run a benchmark on CPU."

    writer.writerow(["Num. particules", "Time (ms)"])
    for np in NUM_PARTICULES:
        print("Running CPU bench (np=%d)" % np)
        writer.writerow([np, run(np, 0, "cpu")])


def GPU_benchmark(writer):
    "Run a benchmark on GPU."

    writer.writerow(["Num. particules", "Block size", "Time (ms)"])
    for np in NUM_PARTICULES:
        for bs in BLOCK_SIZES:
            print("Running GPU bench (np=%d, bs=%d)" % (np, bs))
            writer.writerow([np, bs, run(np, bs, "gpu")])


def run_benchmark(filename, func):
    "Run a benchmark and save results to the given file."

    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        func(writer)


def main():
    "Entry point of this program."

    run_benchmark("cpu.csv", CPU_benchmark)
    run_benchmark("gpu.csv", GPU_benchmark)


if __name__ == "__main__":
    main()
