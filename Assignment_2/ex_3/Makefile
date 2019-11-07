NVCC ?= nvcc
ARCH ?= sm_50

CCFLAGS += -g -O3
CCFLAGS += -arch=$(ARCH) -I/usr/local/cuda/samples/common/inc

exercise_3: exercise_3.cu
	$(NVCC) -o $@ $(CCFLAGS) $<

clean:
	rm exercise_3
