# -Xcompiler " -openmp"

exercice_2:
	nvcc -arch=sm_50 -g -I/usr/local/cuda/samples/common/inc exercise_2.cu -o exercise_2
