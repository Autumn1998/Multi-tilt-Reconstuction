/*
 * FunctionOnGPU.cuh
 *
 *  Created on: Jan 26, 2021
 *      Author: liutong
 */

#ifndef FUNCTIONONGPU_CUH_
#define FUNCTIONONGPU_CUH_
#include "MrcFileIO.h"
#include "ReadAndInitialOptions.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <omp.h>
#include <math.h>
#include <mpi.h>
#include "MrcFileIOBaseFunction.h"
#include "MpiBigData.h"
#define FALSE 0
#define TRUE 1
#define checkCudaErrors( a ) do { \
	if (cudaSuccess != (a)) { \
	fprintf(stderr, "Cuda runtime error in line %d of file %s \
	: %s \n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); \
	exit(EXIT_FAILURE); \
	} \
	} while(0);
void printCudaError();
__device__ void computeWeight(Pixel pixel, int angle,double *x_coef, double *y_coef, Weight *w );
__device__ void BilinearValue(ProjectionSize prj,float *prj_data, Weight wt,double *s,double *c,int ang);
__global__ void backProjOnGPU(ProjectionSize prj,VolumeSize vol,double *x_coef,double *y_coef, float *prj_data, float *model, int slice_start);
__global__ void initialZOnGPU(VolumeSize vol, float *d_w, float *d_w_avg, float *d_z);
__global__ void computeDivisor(ProjectionSize prj,VolumeSize vol,double *d_x_coef,double *d_y_coef,float *d_s,float *d_c,float* d_submodel,int slice_start,int slice_end);	
__device__ void Reproj(ProjectionSize prj,float *d_submodel, Weight wt,float *d_s,float *d_c,int index,int ang);
__global__ void sirtBackProjOnGPU(ProjectionSize prj,VolumeSize vol,double *x_coef,double *y_coef, float *prj_data, float *model, int slice_start,double sirt_step);
__global__ void updateWOnGPU(VolumeSize vol, float *d_w, float *d_v, float *d_z,float step);

int ComputeInitialModel(int i,DataToUse *local_data,InputData *input_data,OutputModel* model,Options *opts);
void runSIRTOnGPU(int i,int leader_id,int group_id,DataToUse *local_data,OutputModel* model,InputData *input_data,Options *opts,MPI_Comm tilt_Comm);
void initialZdata(int device_id,OutputModel *model,Options *opts,float *z,float *w,float *initial_model);
void runMaceOnGPU(int device_id,OutputModel *model,Options *opts,float *v,float *w,float *z,float step);

#endif /* FUNCTIONONGPU_CUH_ */
