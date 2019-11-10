#!/usr/bin/env python3

# On PDC, load Python3 using 'load anaconda/py36/5.0.1'.

import subprocess as sp
import csv
import re
import sys


# Benchmark data.
ARRAY_SIZES = [1000000, 5000000, 10000000, 50000000,
    100000000, 250000000, 500000000, 750000000]
NUM_RUNS = 3

# Regex for matching time.
RE_TIME = re.compile(r"^[GC]PU time: ([0-9]+) ms", re.M)


def run_ps(size, kind, openmp=False):
    "Run the process and return the time."

    prog = ("./exercise_2", "./exercise_2_openmp")[openmp]
    ps = sp.Popen([prog, str(size), kind], stdout=sp.PIPE)

    out, _ = ps.communicate()
    return int(RE_TIME.search(out.decode()).group(1))


def run(*args):
    "Run the benchmark."

    value = sum(run_ps(*args) for _ in range(NUM_RUNS))
    return round(value / NUM_RUNS)


def benchmark(filename, name, args):
    "Run a benchmark with the given args."

    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Array size", "Time (ms)"])

        for size in ARRAY_SIZES:
            print("Running %s bench (size=%d)" % (name, size))
            writer.writerow([size, run(size, *args)])


def main():
    "Entry point of this program."

    benchmark("cpu.csv", "CPU", ["cpu"])
    benchmark("gpu.csv", "GPU", ["gpu"])
    benchmark("cpu-openmp.csv", "CPU-OpenMP", ["cpu-openmp", True])


if __name__ == "__main__":
    main()
