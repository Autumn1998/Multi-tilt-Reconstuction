#include "FunctionOnGPU.cuh"

void printCudaError()
{
	cudaError_t cudaStatus = cudaGetLastError();
	if(cudaStatus!=cudaSuccess)
	{
		printf("Kernel lauch failed:%s\n",cudaGetErrorString(cudaStatus));
	}
}

__device__ void computeWeight(Pixel pixel, int angle,double *x_coef, double *y_coef, Weight *w )
{
   double x,y;

   int index=4*angle;
   x=x_coef[index]+x_coef[index+1]*pixel.X+x_coef[index+2]*pixel.Y+x_coef[index+3]*pixel.Z -1;
   y=y_coef[index]+y_coef[index+1]*pixel.X+y_coef[index+2]*pixel.Y+y_coef[index+3]*pixel.Z -1;
   w->x_min=floor(x);
   w->y_min=floor(y);

   w->x_min_del = x - w->x_min;
   w->y_min_del = y - w->y_min;
  // if(a == 1)printf("Compute : `x_min:%lf   y_Min:%lf\n",w->x_min_del,w->y_min_del);
}

__device__ void Reproj(ProjectionSize prj,float *d_submodel, Weight wt,float *d_s,float *d_c,int index,int ang)
{
  	int n;
	//if(a==1) printf("Oringinal a:%lf \n",*c);
	if(wt.x_min >= 0 && wt.x_min < prj.X && wt.y_min >= 0 && wt.y_min < prj.Y){ //(x_min, y_min)
		n = wt.x_min + wt.y_min * prj.X + ang*prj.X*prj.Y;
		atomicAdd(&d_s[n],(1-wt.x_min_del) * (1-wt.y_min_del)*d_submodel[index]);
		atomicAdd(&d_c[n],(1-wt.x_min_del) * (1-wt.y_min_del));
	}
	if((wt.x_min+1) >= 0 && (wt.x_min+1) < prj.X && wt.y_min >= 0 && wt.y_min < prj.Y){ //(x_min+1, y_min)
		n = wt.x_min+1 + wt.y_min * prj.X + ang*prj.X*prj.Y;
		atomicAdd(&d_s[n],wt.x_min_del * (1-wt.y_min_del)*d_submodel[index]);
		atomicAdd(&d_c[n],wt.x_min_del * (1-wt.y_min_del));
	}
	if(wt.x_min >= 0 && wt.x_min < prj.X && (wt.y_min+1) >= 0 && (wt.y_min+1) < prj.Y){ //(x_min, y_min+1)
		n = wt.x_min + (wt.y_min+1) * prj.X + ang*prj.X*prj.Y;
		atomicAdd(&d_s[n],(1-wt.x_min_del) * wt.y_min_del*d_submodel[index]);
		atomicAdd(&d_c[n],(1-wt.x_min_del) * wt.y_min_del);
	}
	if((wt.x_min+1) >= 0 && (wt.x_min+1) < prj.X && (wt.y_min+1) >= 0 && (wt.y_min+1) < prj.Y){ //(x_min+1, y_min+1)
		n = wt.x_min+1 + (wt.y_min+1) * prj.X + ang*prj.X*prj.Y;
		atomicAdd(&d_s[n],wt.x_min_del * wt.y_min_del*d_submodel[index]);
		atomicAdd(&d_c[n],wt.x_min_del * wt.y_min_del);
	}
//	if(a == 1)printf("prj.x:%d  prj.Y:%d  x_min:%d   y_min:%d   x_min_del:%lf   y_min_del:%lf  c:%lf \n",prj.X,prj.Y,wt.x_min,wt.y_min,wt.x_min_del,wt.y_min_del,*c);
}

__device__ void BilinearValue(ProjectionSize prj,float *prj_data, Weight wt,double *s,double *c,int ang)
{
  	int n;
	//if(a==1) printf("Oringinal a:%lf \n",*c);
	if(wt.x_min >= 0 && wt.x_min < prj.X && wt.y_min >= 0 && wt.y_min < prj.Y){ //(x_min, y_min)
		n = wt.x_min + wt.y_min * prj.X + ang*prj.X*prj.Y;
		*s += (1-wt.x_min_del) * (1-wt.y_min_del) * prj_data[n];
		*c += (1-wt.x_min_del) * (1-wt.y_min_del);
	}
	if((wt.x_min+1) >= 0 && (wt.x_min+1) < prj.X && wt.y_min >= 0 && wt.y_min < prj.Y){ //(x_min+1, y_min)
		n = wt.x_min+1 + wt.y_min * prj.X + ang*prj.X*prj.Y;
		*s += wt.x_min_del * (1-wt.y_min_del) * prj_data[n];
		*c += wt.x_min_del * (1-wt.y_min_del);
	}
	if(wt.x_min >= 0 && wt.x_min < prj.X && (wt.y_min+1) >= 0 && (wt.y_min+1) < prj.Y){ //(x_min, y_min+1)
		n = wt.x_min + (wt.y_min+1) * prj.X + ang*prj.X*prj.Y;
		*s += (1-wt.x_min_del) * wt.y_min_del * prj_data[n];
		*c += (1-wt.x_min_del) * wt.y_min_del;
	}
	if((wt.x_min+1) >= 0 && (wt.x_min+1) < prj.X && (wt.y_min+1) >= 0 && (wt.y_min+1) < prj.Y){ //(x_min+1, y_min+1)
		n = wt.x_min+1 + (wt.y_min+1) * prj.X + ang*prj.X*prj.Y;
		*s += wt.x_min_del * wt.y_min_del * prj_data[n];
		*c += wt.x_min_del * wt.y_min_del;
	}
//	if(a == 1)printf("prj.x:%d  prj.Y:%d  x_min:%d   y_min:%d   x_min_del:%lf   y_min_del:%lf  c:%lf \n",prj.X,prj.Y,wt.x_min,wt.y_min,wt.x_min_del,wt.y_min_del,*c);
}


__global__ void backProjOnGPU(ProjectionSize prj,VolumeSize vol,double *x_coef,double *y_coef, float *prj_data, float *model, int slice_start, int slice_end)
{
    double s = 0;//分子
    double c = 0;//分母
    int x = threadIdx.x+blockIdx.x*blockDim.x +vol.Xstart;
    int y = threadIdx.y+blockIdx.y*blockDim.y +vol.Ystart;
    int z = threadIdx.z+blockIdx.z*blockDim.z +slice_start;
    //if(x == 30 && y ==100 && z == 1+slice_start) printf("startartatratrtartatrtartartatr\n");
    if(x>=vol.Xend || y>=vol.Yend ||z>=slice_end) return;
    Pixel p;
    p.X =x;p.Y=y;p.Z=z;
    
    int slice_index=(x-vol.Xstart)+(y-vol.Ystart)*vol.X+(z-slice_start)*vol.X*vol.Y;
    Weight w;

    for(int angle=0;angle<prj.AngN;angle++)
    {
        computeWeight(p,angle,x_coef,y_coef,&w);
        BilinearValue(prj,prj_data,w,&s,&c,angle);
    }
    //if(x == 30 && y ==100 && z == 1+slice_start) printf("slice_index:%d  index:%d  (%f %f %f)  ON GPU\n",slice_index,index,s,c,s/c);
    if(c!=0.0f) 
    {
      //  if(slice_index>=vol.X*vol.Y*(slice_end-slice_start)) printf("slice_index:%d  upbound:%d  now X:%d  Y:%d  Z:%d     ->   volstart.X:%d Y:%d Z:%d    volend.X:%d Y:%d Z:%d   vol.X:%d Y:%d Z:%d\n",
      //  slice_index,vol.X*vol.Y*(slice_end-slice_start),x,y,z,vol.Xstart,vol.Ystart,slice_start,vol.Xend,vol.Yend,slice_end,vol.X,vol.Y,(slice_end-slice_start)); return;
        model[slice_index] += (float)(s/c);
    }
}

__global__ void sirtBackProjOnGPU(ProjectionSize prj,VolumeSize vol,double *x_coef,double *y_coef, float *prj_data, float *model, int slice_start, int slice_end,double sirt_step)
{
    double s = 0;//分子
    double c = 0;//分母
    int x = threadIdx.x+blockIdx.x*blockDim.x +vol.Xstart;
    int y = threadIdx.y+blockIdx.y*blockDim.y +vol.Ystart;
    int z = threadIdx.z+blockIdx.z*blockDim.z +slice_start;
    //if(x == 30 && y ==100 && z == 1+slice_start) printf("startartatratrtartatrtartartatr\n");
    if(x>=vol.Xend || y>=vol.Yend ||z>=slice_end) return;
    Pixel p;
    p.X =x;p.Y=y;p.Z=z;
    
    int slice_index=(x-vol.Xstart)+(y-vol.Ystart)*vol.X+(z-slice_start)*vol.X*vol.Y;
    Weight w;

    for(int angle=0;angle<prj.AngN;angle++)
    {
        computeWeight(p,angle,x_coef,y_coef,&w);
        BilinearValue(prj,prj_data,w,&s,&c,angle);
    }
    if(c!=0.0f) 
    {
        model[slice_index] += (float)(s/c)*sirt_step;
    }
}


int ComputeInitialModel(int i,DataToUse *local_data,InputData *input_data,OutputModel* model,Options *opts)
{
    cudaSetDevice(i%4);
    //getGPUInfo(i);
    //printf(">>>>>>>>>>>>>  BPT Runing ON Thread ID( Device ID):%d ... \n",i);
    double *d_x_coef,*d_y_coef;
    float *d_mrc_data,*d_submodel;
    long long coef_size = sizeof(double *)*local_data->angN*10;
    long long mrc_size = sizeof(float)*input_data->prj.X*input_data->prj.Y*local_data->angN;
    int z_thickness = model->slc.z_end - model->slc.z_start;
    long long subvol_size = sizeof(float)*model->data_size;
    //long long offset = sizeof(float)*input_data->vol.X*input_data->vol.Y*(slices[i].z_start-input_data->vol.Zstart);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    //printf("%d : prj-> %d %d %d   vol->%d %d %d\n",i,input_data->prj.X,input_data->prj.Y,input_data->prj.AngN,input_data->vol.X,input_data->vol.Y,z_thickness);

    checkCudaErrors(cudaMalloc((void**)&d_x_coef, coef_size));
    checkCudaErrors(cudaMalloc((void**)&d_y_coef, coef_size));
    checkCudaErrors(cudaMalloc((void**)&d_mrc_data, mrc_size));
    checkCudaErrors(cudaMalloc((void**)&d_submodel, subvol_size));
    checkCudaErrors(cudaMemcpyAsync(d_x_coef,local_data->x_coef,coef_size,cudaMemcpyHostToDevice,stream));
    checkCudaErrors(cudaMemcpyAsync(d_y_coef,local_data->y_coef,coef_size,cudaMemcpyHostToDevice,stream));
    checkCudaErrors(cudaMemcpyAsync(d_mrc_data,local_data->mrc_data,mrc_size,cudaMemcpyHostToDevice,stream));
    
    dim3 block(opts->block_x,opts->block_y,opts->block_z);
    dim3 grid((model->vol.X+block.x-1)/block.x,(model->vol.Y+block.y-1)/block.y,(z_thickness+block.z-1)/block.z);
    backProjOnGPU<<<grid,block,0,stream>>>(input_data->prj,input_data->vol,d_x_coef,d_y_coef,d_mrc_data,d_submodel,model->slc.z_start,model->slc.z_end);
    printCudaError();

    cudaDeviceSynchronize();
   // printf("On ID:%d  model->vol.X*model->vol.Y*z_thinkness:%lld \n",i,sizeof(float)*model->vol.X*model->vol.Y*z_thickness);
    checkCudaErrors(cudaMemcpyAsync(model->output_data,d_submodel,subvol_size,cudaMemcpyDeviceToHost,stream));
//    printf("1204685 : %f  at  %d\n",buffer[i][1204685],i);
    cudaFree(d_x_coef);
    cudaFree(d_y_coef);
    cudaFree(d_mrc_data);
    cudaFree(d_submodel);
    cudaStreamDestroy(stream);
    //printf(">>>>>>>>>>>>>  BPT finished at Thread %d \n",i);
    
    return 1;
}


void runSIRTOnGPU(int i,int leader_id, int group_id,DataToUse *local_data,OutputModel* model,InputData *input_data,Options *opts,MPI_Comm tilt_comm )
{
    cudaSetDevice(i%4);
    double *d_x_coef,*d_y_coef;
    float *d_submodel,*d_s,*d_c;
	float *h_s,*h_c,*s_buffer,*c_buffer;
    long long coef_size = sizeof(double *)*local_data->angN*10;
    long long mrc_size = sizeof(float)*input_data->prj.X*input_data->prj.Y*local_data->angN;
    int z_thickness = model->slc.z_end - model->slc.z_start;
    long long subvol_size = sizeof(float)*model->data_size;
   
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
	h_s = (float *)malloc(mrc_size);
	h_c = (float *)malloc(mrc_size);
	s_buffer = (float *)malloc(mrc_size);
	c_buffer = (float *)malloc(mrc_size);
	checkCudaErrors(cudaMalloc((void**)&d_x_coef, coef_size));
    checkCudaErrors(cudaMalloc((void**)&d_y_coef, coef_size));
    checkCudaErrors(cudaMalloc((void**)&d_s, mrc_size));
    checkCudaErrors(cudaMalloc((void**)&d_c, mrc_size));
    checkCudaErrors(cudaMalloc((void**)&d_submodel, subvol_size));
    checkCudaErrors(cudaMemcpyAsync(d_x_coef,local_data->x_coef,coef_size,cudaMemcpyHostToDevice,stream));
    checkCudaErrors(cudaMemcpyAsync(d_y_coef,local_data->y_coef,coef_size,cudaMemcpyHostToDevice,stream));
    checkCudaErrors(cudaMemcpyAsync(d_submodel,model->output_data,subvol_size,cudaMemcpyHostToDevice,stream));
    
    dim3 block(opts->block_x,opts->block_y,opts->block_z);
    dim3 grid((model->vol.X+block.x-1)/block.x,(model->vol.Y+block.y-1)/block.y,(z_thickness+block.z-1)/block.z);
   
    for(int cnt=0;cnt<10;cnt++)
	{
	memset(h_s,0,mrc_size);
	memset(h_c,0,mrc_size);
    CHECK(cudaMemcpyAsync(d_s,h_s,mrc_size,cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemcpyAsync(d_c,h_c,mrc_size,cudaMemcpyHostToDevice,stream));
		computeDivisor<<<grid,block,0,stream>>>(input_data->prj,input_data->vol,d_x_coef,d_y_coef,d_s,d_c,d_submodel,model->slc.z_start,model->slc.z_end);
		printCudaError();

	    checkCudaErrors(cudaMemcpyAsync(h_c,d_c,mrc_size,cudaMemcpyDeviceToHost,stream));
	    checkCudaErrors(cudaMemcpyAsync(h_s,d_s,mrc_size,cudaMemcpyDeviceToHost,stream));
		cudaDeviceSynchronize();

		MPI_Reduce_BIG(h_c,c_buffer, mrc_size/sizeof(float) ,MPI_FLOAT,MPI_SUM,0,tilt_comm); 
		MPI_Reduce_BIG(h_s,s_buffer, mrc_size/sizeof(float) ,MPI_FLOAT,MPI_SUM,0,tilt_comm); 

		if(i==leader_id)
		{
			for(int k=0;k<mrc_size/sizeof(float);k++)
			{
				if(c_buffer[k]!=0) s_buffer[k]/=c_buffer[k];
				s_buffer[k] = local_data->mrc_data[k]-s_buffer[k];
			}
		
		}
		MPI_Bcast_BIG(s_buffer,mrc_size/sizeof(float) ,MPI_FLOAT,0,tilt_comm);
		MPI_Barrier(tilt_comm);
	
	    checkCudaErrors(cudaMemcpyAsync(d_s,s_buffer,mrc_size,cudaMemcpyHostToDevice,stream));
	    sirtBackProjOnGPU<<<grid,block,0,stream>>>(input_data->prj,input_data->vol,d_x_coef,d_y_coef,d_s,d_submodel,model->slc.z_start,model->slc.z_end,opts->sirt_step);
		printCudaError();
	    cudaDeviceSynchronize();
	}

    checkCudaErrors(cudaMemcpyAsync(model->output_data,d_submodel,subvol_size,cudaMemcpyDeviceToHost,stream));
   // printf("On ID:%d  model->vol.X*model->vol.Y*z_thinkness:%lld \n",i,sizeof(float)*model->vol.X*model->vol.Y*z_thickness);
//    printf("1204685 : %f  at  %d\n",buffer[i][1204685],i);
    cudaFree(d_c);
    free(h_s);
	free(h_c);
	free(c_buffer);
	h_s=NULL;
	h_c=NULL;
	c_buffer=NULL;

    cudaFree(d_x_coef);
    cudaFree(d_y_coef);
	cudaFree(d_s);
    cudaFree(d_submodel);
    cudaStreamDestroy(stream);
	free(s_buffer);
	s_buffer=NULL;
    //printf(">>>>>>>>>>>>>  BPT finished at Thread %d \n",i);
    
}

__global__ void initialZOnGPU(VolumeSize vol,float *d_w,float *d_w_avg,float *d_z)
{
	int x = threadIdx.x+blockIdx.x*blockDim.x +vol.Xstart;
    int y = threadIdx.y+blockIdx.y*blockDim.y +vol.Ystart;
    int z = threadIdx.z+blockIdx.z*blockDim.z +vol.Zstart;
    if(x>=vol.Xend || y>=vol.Yend ||z>=vol.Zend) return;
	int i = (x-vol.Xstart)+(y-vol.Ystart)*vol.X+(z-vol.Zstart)*vol.X*vol.Y;
	d_z[i] = 2*d_w_avg[i] - d_w[i];
}

__global__ void updateWOnGPU(VolumeSize vol,float *d_v,float *d_w,float *d_z,float step)
{
	int x = threadIdx.x+blockIdx.x*blockDim.x +vol.Xstart;
    int y = threadIdx.y+blockIdx.y*blockDim.y +vol.Ystart;
    int z = threadIdx.z+blockIdx.z*blockDim.z +vol.Zstart;
    if(x>=vol.Xend || y>=vol.Yend ||z>=vol.Zend) return;
	int i = (x-vol.Xstart)+(y-vol.Ystart)*vol.X+(z-vol.Zstart)*vol.X*vol.Y;
    if(i==20000){printf("On GPU before 20000: -> dw[20000]:%f\n",d_w[i]);}
	d_w[i] = step*(2*d_v[i]-d_z[i]) + (1-step)*d_w[i];
    if(i==20000){printf("On GPU after 20000: -> dw[20000]:%f\n",d_w[i]);}
}


void initialZdata(int device_id,OutputModel *model,Options *opts,float *z,float *w,float *initial_model)
{
    cudaSetDevice(device_id%4);
    dim3 block(opts->block_x,opts->block_y,opts->block_z);
    dim3 grid((model->vol.X+block.x-1)/block.x,(model->vol.Y+block.y-1)/block.y,(model->vol.Z+block.z-1)/block.z);
    long long vol_size = sizeof(float)*model->vol.X*model->vol.Y*model->vol.Z;
	float *d_w,*d_w_avg,*d_z;
    checkCudaErrors(cudaMalloc((void**)&d_w, vol_size));
    checkCudaErrors(cudaMalloc((void**)&d_w_avg, vol_size));
    checkCudaErrors(cudaMalloc((void**)&d_z, vol_size));
	cudaStream_t stream;
    cudaStreamCreate(&stream);
	checkCudaErrors(cudaMemcpyAsync(d_w,w,vol_size,cudaMemcpyHostToDevice,stream));
    checkCudaErrors(cudaMemcpyAsync(d_w_avg,initial_model,vol_size,cudaMemcpyHostToDevice,stream)); 
	initialZOnGPU<<<grid,block>>>(model->vol,d_w,d_w_avg,d_z);	
    printCudaError();
	checkCudaErrors(cudaMemcpyAsync(z,d_z,vol_size,cudaMemcpyDeviceToHost,stream));
    cudaDeviceSynchronize();
    cudaFree(d_w);
    cudaFree(d_w_avg);
    cudaFree(d_z);
    cudaStreamDestroy(stream);
}


__global__ void computeDivisor(ProjectionSize prj,VolumeSize vol,double *d_x_coef,double *d_y_coef,float *d_s,float *d_c,float* d_submodel,int slice_start,int slice_end)
{
    int x = threadIdx.x+blockIdx.x*blockDim.x +vol.Xstart;
    int y = threadIdx.y+blockIdx.y*blockDim.y +vol.Ystart;
    int z = threadIdx.z+blockIdx.z*blockDim.z +slice_start;
    if(x>=vol.Xend || y>=vol.Yend ||z>=slice_end) return;
    Pixel p;
    p.X =x;p.Y=y;p.Z=z;
    
    int slice_index=(x-vol.Xstart)+(y-vol.Ystart)*vol.X+(z-slice_start)*vol.X*vol.Y;
    Weight w;
    
    for(int angle=0;angle<prj.AngN;angle++)
    {
        computeWeight(p,angle,d_x_coef,d_y_coef,&w);
        Reproj(prj,d_submodel,w,d_s,d_c,slice_index,angle);
    }
}


void runMaceOnGPU(int device_id,OutputModel *model,Options *opts,float *v,float *w,float *z,float step)
{
	cudaSetDevice(device_id%4);
    dim3 block(opts->block_x,opts->block_y,opts->block_z);
    dim3 grid((model->vol.X+block.x-1)/block.x,(model->vol.Y+block.y-1)/block.y,(model->vol.Z+block.z-1)/block.z);
    long long vol_size = sizeof(float)*model->vol.X*model->vol.Y*model->vol.Z;
	float *d_w,*d_v,*d_z;
    checkCudaErrors(cudaMalloc((void**)&d_w, vol_size));
    printf("ID:%d Just before GPU W[20000]:%f \n",device_id,w[20000]);
    checkCudaErrors(cudaMalloc((void**)&d_v, vol_size));
    checkCudaErrors(cudaMalloc((void**)&d_z, vol_size));
	cudaStream_t stream;
    cudaStreamCreate(&stream);
	checkCudaErrors(cudaMemcpyAsync(d_w,w,vol_size,cudaMemcpyHostToDevice,stream));
    checkCudaErrors(cudaMemcpyAsync(d_v,v,vol_size,cudaMemcpyHostToDevice,stream)); 
    checkCudaErrors(cudaMemcpyAsync(d_z,z,vol_size,cudaMemcpyHostToDevice,stream)); 
	updateWOnGPU<<<grid,block>>>(model->vol,d_w,d_v,d_z,step);	
    printCudaError();
	checkCudaErrors(cudaMemcpyAsync(w,d_w,vol_size,cudaMemcpyDeviceToHost,stream));
    cudaDeviceSynchronize();
    cudaFree(d_w);
    printf("ID:%d Just off GPU W[20000]:%f \n",device_id,w[20000]);
    cudaFree(d_v);
    cudaFree(d_z);
    cudaStreamDestroy(stream);
	
}


