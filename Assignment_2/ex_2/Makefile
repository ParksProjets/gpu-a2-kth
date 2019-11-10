NVCC ?= nvcc
GCC ?= g++

ifeq ($(MAKECMDGOALS),pdc)
  ARCH ?= sm_37
else
  ARCH ?= sm_50
endif


CCFLAGS += -g
NVCCFLAGS += -arch=$(ARCH) -I/usr/local/cuda/samples/common/inc

.PHONY: all pdc clean


all: exercise_2 exercise_2_openmp

exercise_2: exercise_2.cu
	$(NVCC) -o $@ $(CCFLAGS) $(NVCCFLAGS) $<

exercise_2_openmp: exercise_2_openmp.cc
	$(GCC) -fopenmp -o $@ $(CCFLAGS) $<


pdc: exercise_2 exercise_2_openmp benchmark.py
	cp $^ /cfs/klemming/nobackup/$(shell expr substr $(USER) 1 1)/$(USER)

clean:
	rm -f exercise_2 exercise_2_openmp
