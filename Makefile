objects= FunctionOnGPU.o MultiAxis.o MrcFileIO.o ReadAndInitialOptions.o MrcFileIOBaseFunction.o MpiBigData.o
MAIN = MultiAxis.cpp MrcFileIO.cpp ReadAndInitialOptions.cpp MrcFileIOBaseFunction.cpp MpiBigData.cpp
CUAFLAGS = -Xcompiler -fopenmp -lcuda  #-gencode arch=compute_61,code=sm_61 #-arch=sm_35
MPIFLAGS = -L/GPUFS/app_GPU/compiler/CUDA/10.1.2/lib64 -lcudart

MultiAxis:$(objects)
	mpic++ $(MPIFLAGS) -o MultiAxis $(objects) 

MrcFileIO.o:$(MAIN) 
	nvcc $(CUAFLAGS) -c  $(MAIN)

FunctionOnGPU.o: FunctionOnGPU.cu
	nvcc $(CUAFLAGS) -c  FunctionOnGPU.cu

.PHONY:clean
clean:
	rm MultiAxis $(objects)
