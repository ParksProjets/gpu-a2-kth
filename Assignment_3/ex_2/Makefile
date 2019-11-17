NVCC ?= nvcc

ifeq ($(MAKECMDGOALS),pdc)
  ARCH ?= sm_37
else
  ARCH ?= sm_50
endif


CCFLAGS += -g -O3
CCFLAGS += -arch=$(ARCH) -I/usr/local/cuda/samples/common/inc

.PHONY: all pdc clean


all: exercise_2a exercise_2b

exercise_2a: exercise_2a.cu
	$(NVCC) -o $@ $(CCFLAGS) $<

exercise_2b: exercise_2b.cu
	$(NVCC) -o $@ $(CCFLAGS) $<


pdc: exercise_2a exercise_2b
	cp $^ /cfs/klemming/nobackup/$(shell expr substr $(USER) 1 1)/$(USER)

clean:
	rm -f exercise_2a exercise_2b
