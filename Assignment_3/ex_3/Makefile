NVCC ?= nvcc

ifeq ($(MAKECMDGOALS),pdc)
  ARCH ?= sm_37
else
  ARCH ?= sm_50
endif


CCFLAGS += -g -O3 -lcurand -lcublas
CCFLAGS += -arch=$(ARCH) -I/usr/local/cuda/samples/common/inc

.PHONY: pdc clean

exercise_3: exercise_3.cu
	$(NVCC) -o $@ $< $(CCFLAGS)

pdc: exercise_3 benchmark.py
	cp $^ /cfs/klemming/nobackup/$(shell expr substr $(USER) 1 1)/$(USER)

clean:
	rm -f exercise_3
