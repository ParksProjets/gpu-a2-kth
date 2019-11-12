NVCC ?= nvcc
ARCH ?= sm_50

CCFLAGS += -g -O3
CCFLAGS += -arch=$(ARCH) -I/usr/local/cuda/samples/common/inc

exercice_1: exercice_1.cu
	$(NVCC) -o $@ $(CCFLAGS) $<

clean:
	rm exercice_1
