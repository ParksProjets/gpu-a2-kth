#!/usr/bin/env python3

# On PDC, load Python3 using 'load anaconda/py36/5.0.1'.

import subprocess as sp
import csv
import re
import sys


# Benchmark data.
NUM_PARTICULES = [
    100, 1000, 5000, 10000, 50000, 100000, 500000,
    1000000, 2500000, 4000000, 6000000, 8000000,
    10000000
]

BLOCK_SIZES = [16, 32, 64, 80, 100, 128, 192, 256]
NUM_ITERATIONS = 1000

NUM_RUNS = 3


# Regex for matching time.
RE_TIME = re.compile(r"^[GC]PU time: ([0-9]+) ms", re.M)


def run_ps(num_particules, block_size, kind):
    "Run the process and return the time."

    args = [num_particules, NUM_ITERATIONS, block_size, kind]
    args = [str(v) for v in args]

    ps = sp.Popen(["./exercise_3", *args], stdout=sp.PIPE)

    out, _ = ps.communicate()
    return int(RE_TIME.search(out.decode()).group(1))


def run(*args):
    "Run the benchmark."

    value = sum(run_ps(*args) for _ in range(NUM_RUNS))
    return round(value / NUM_RUNS)


def CPU_benchmark(writer):
    "Run a benchmark on CPU."

    writer.writerow(["Num. particules", "Time (ms)"])
    for np in NUM_PARTICULES:
        print("Running CPU bench (np=%d)" % np)
        writer.writerow([np, run(np, 0, "cpu")])


def GPU_benchmark(writer):
    "Run a benchmark on GPU."

    writer.writerow(["Block size", "Num. particules", "Time (ms)"])
    for bs in BLOCK_SIZES:
        for np in NUM_PARTICULES:
            print("Running GPU bench (bs=%d, np=%d)" % (bs, np))
            writer.writerow([bs, np, run(np, bs, "gpu")])


def run_benchmark(filename, func):
    "Run a benchmark and save results to the given file."

    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        func(writer)


def main():
    "Entry point of this program."

    # run_benchmark("cpu.csv", CPU_benchmark)
    run_benchmark("gpu.csv", GPU_benchmark)


if __name__ == "__main__":
    main()
