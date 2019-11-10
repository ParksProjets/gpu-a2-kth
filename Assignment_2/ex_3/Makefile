NVCC ?= nvcc
ARCH ?= sm_50

ifeq ($(MAKECMDGOALS),pdc)
  ARCH = sm_30
endif


CCFLAGS += -g -O3
CCFLAGS += -arch=$(ARCH) -I/usr/local/cuda/samples/common/inc

.PHONY: pdc clean


exercise_3: exercise_3.cu
	$(NVCC) -o $@ $(CCFLAGS) $<

pdc: exercise_3 benchmark.py
	cp $^ /cfs/klemming/nobackup/$(shell expr substr $(USER) 1 1)/$(USER)

clean:
	rm exercise_3
