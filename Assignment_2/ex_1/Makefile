NVCC ?= nvcc
ARCH ?= sm_50

CCFLAGS += -g -Og
CCFLAGS += -arch=$(ARCH) -I/usr/local/cuda/samples/common/inc

exercice_1:
	$(NVCC) -o $@ $(CCFLAGS) $<

clean:
	rm exercice_1
