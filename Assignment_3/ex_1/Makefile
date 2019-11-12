NVCC ?= nvcc
ARCH ?= sm_50

CCFLAGS += -g -O3
CCFLAGS += -arch=$(ARCH) -I/usr/local/cuda/samples/common/inc

hw3_ex1: hw3_ex1.cu
	$(NVCC) -o $@ $(CCFLAGS) $<

clean:
	rm hw3_ex1
