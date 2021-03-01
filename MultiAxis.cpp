#include "FunctionOnGPU.cuh"
#include "ReadAndInitialOptions.h"
#include "MrcFileIO.h"
#include "mpi.h"
#include "MpiBigData.h"
#include <iostream>
#include <time.h>
//#include "nccl.h"
#define TXBR_LINE_NUM 4 //TXBR每行有几个数据
using namespace std;


Slice divideSlice(int myid,int N,int Z,int S)
{
    Slice slice;
    int zrem = Z%N;//余数
    //printf("*********DIVIDE***********************\n          Z_start      Z_end   \n");
    int height;
    if(myid < zrem){
        height = Z/N+1;
        slice.z_start = height *myid+S;
    }
    else{
        height = Z/N;
        slice.z_start = height *myid+zrem+S;
    }
    slice.z_end = slice.z_start+height;
    //printf("Slice %d: |%-10d   %-10d|\n",myid,slice.z_start,slice.z_end);
    //printf("*************************************\n");
    return slice;
}

void runMaceOnCPU(int myid,float *v,float *w,float *z,int data_size,float step)
{
    if(data_size<0) printf("int over flow!\n");
    //printf("ID:%d, before step:%f  W【10000】:%f z[10000]:%f V[10000]:%f, res:%f,+ %f\n",myid,step,w[k],z[k],v[k],step*(2*v[k]-z[k]),(1-step)*w[k]);
    for(int i=0;i<data_size;i++) w[i] = step*(2*v[i]-z[i])+(1-step)*w[i];
}

void reconstructModel(int myid,int numprocs,float *w,float *z,DataToUse *local_data,InputData *input_data,float *w_avg,OutputModel *model,Options *opts,int *send_count,int *displs, int group_id, MPI_Comm tilt_comm,int leader_id)
{
	int N = opts->iter_times;
	int s = opts->sirt_step;
	float *v = (float *)malloc(sizeof(float)*model->vol.X*model->vol.Y*model->vol.Z);
	for(int i=0;i<N;i++)
	{
		//initail z
		if(myid==leader_id)
			initialZdata(myid,model,opts,z,w,w_avg);
		MPI_Scatterv(z,send_count,displs,MPI_FLOAT,model->output_data,model->data_size,MPI_FLOAT,0,tilt_comm);
		runSIRTOnGPU(myid,leader_id,group_id,local_data,model,input_data,opts,tilt_comm);
		MPI_Gatherv(model->output_data,model->data_size,MPI_FLOAT,v,send_count,displs,MPI_FLOAT,0,tilt_comm);
		//checkdata(myid,z,model->vol.X*model->vol.Y*model->vol.Z);
 //       printf("ID:%d,  step_sirt:%f  mace:%f after mace   W【10000】:%f z[10000]:%f V[10000]:%f\n",myid,opts->sirt_step,opts->mace_step,w[10000],z[10000],v[10000]);
		if(myid==leader_id)
		{
			//runMaceOnGPU(myid,model,opts,v,w,z,opts->mace_step);	
			runMaceOnCPU(myid,v,w,z,model->vol.X*model->vol.Y*model->vol.Z,opts->mace_step);	
		}else memset(w,0,sizeof(float)*model->vol.X*model->vol.Y*model->vol.Z);
  //      printf("ID:%d,  after mace   W[10000]:%f\n",myid,w[10000]);
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Reduce_BIG(w,w_avg,(long long)model->vol.X*model->vol.Y*model->vol.Z ,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
		if(myid==0) for(int k=0;k<model->vol.X*model->vol.Y*model->vol.Z; k++) w_avg[k]/=numprocs;
		MPI_Bcast_BIG(w_avg,model->vol.X*model->vol.Y*model->vol.Z,MPI_FLOAT,0,MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
      //  printf("ID:%d,  after iter   W_avg[10000]:%f\n",myid,w_avg[10000]);
		if(myid==0) printf("Iteration %d finished\n",i);
	}
}

void synchronizeVol(InputData *input_data)
{
    int buf;
    MPI_Reduce(&input_data->vol.Xstart,&buf,1,MPI_INT,MPI_MIN,0,MPI_COMM_WORLD);
    input_data->vol.Xstart = buf;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&input_data->vol.Ystart,&buf,1,MPI_INT,MPI_MIN,0,MPI_COMM_WORLD);
    input_data->vol.Ystart = buf;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&input_data->vol.Zstart,&buf,1,MPI_INT,MPI_MIN,0,MPI_COMM_WORLD);
    input_data->vol.Zstart = buf;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&input_data->vol.Xend,&buf,1,MPI_INT,MPI_MAX,0,MPI_COMM_WORLD);
    input_data->vol.Xend = buf;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&input_data->vol.Yend,&buf,1,MPI_INT,MPI_MAX,0,MPI_COMM_WORLD);
    input_data->vol.Yend = buf;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&input_data->vol.Zend,&buf,1,MPI_INT,MPI_MAX,0,MPI_COMM_WORLD);
    input_data->vol.Zend = buf;
    MPI_Barrier(MPI_COMM_WORLD);   
    MPI_Bcast(&input_data->vol,sizeof(VolumeSize)/sizeof(int),MPI_INT,0,MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);   
    input_data->vol.X = input_data->vol.Xend-input_data->vol.Xstart;
    input_data->vol.Y = input_data->vol.Yend-input_data->vol.Ystart;
    input_data->vol.Z = input_data->vol.Zend-input_data->vol.Zstart;
}

int main(int argc,char *argv[])
{
    //ncclComm_t comms[1];
    Options opts;
    InputData input_data;
    DataToUse local_data;

    OutputModel model;
	long long vol_data_size;
    int myid, numprocs,N,group_id,leader_id;//N - 每轴几个线程
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

//**********读取参数和数据
    opts.slice_num=numprocs;
    if(readOptions(argc, argv, &opts) < 0) {
        printf("***WRONG INPUT.\n");
//            exit(-1);
    }

   //将线程均分给每个轴  线程分组
    N = numprocs/opts.number_tilt;
    group_id=myid/N;
    MPI_Comm tilt_comm;
    MPI_Comm_split(MPI_COMM_WORLD,group_id,myid,&tilt_comm);
	leader_id = group_id*N;

    input_data.thread_id = myid;
    input_data.group_id = group_id;
    if(myid==leader_id)
    {
        if(numprocs%opts.number_tilt!=0){
            printf("***线程数必须是轴数的倍数.\n");
//            exit(-1);
        }
        if(readMrcHeader(&opts, &input_data) < 0) {
            printf("***WRONG AT READ HEADER FILE.\n");
            return -1;
        }
    }
    MPI_Bcast(&input_data,sizeof(InputData)/sizeof(char),MPI_CHAR,0,tilt_comm);
	long long prj_data_size = (long long)input_data.prj.X*input_data.prj.Y*input_data.prj.AngN;
//	printf("prj_data size : %lld  %d %d %d\n",prj_data_size,input_data.prj.X,input_data.prj.Y,input_data.prj.AngN);
    local_data.mrc_data =  (float *)malloc(sizeof(float)*prj_data_size);
    local_data.x_coef =  (double *)malloc(sizeof(double)*input_data.prj.AngN*10);
    local_data.y_coef =  (double *)malloc(sizeof(double)*input_data.prj.AngN*10);
    if(myid == leader_id)
	{
		if(readMrcAndTXBR(&opts, &input_data, &local_data)<0)
            printf("***Wrong INPUT\n");
	}
    
    MPI_Bcast_BIG(local_data.mrc_data,prj_data_size,MPI_FLOAT,0,tilt_comm);
    MPI_Bcast(local_data.x_coef,input_data.prj.AngN*10,MPI_DOUBLE,0,tilt_comm);
    MPI_Bcast(local_data.y_coef,input_data.prj.AngN*10,MPI_DOUBLE,0,tilt_comm);
    MPI_Barrier(MPI_COMM_WORLD);
    synchronizeVol(&input_data);
	//printf("ID:%d st:%d %d %d ed:%d %d %d x:%d y:%d z:%d \n",myid,input_data.vol.Xstart,input_data.vol.Ystart,input_data.vol.Zstart,input_data.vol.Xend,input_data.vol.Yend,input_data.vol.Zend,input_data.vol.X,input_data.vol.Y,input_data.vol.Z);
// ***********


    model.slc = divideSlice(myid%N,N,input_data.vol.Z,input_data.vol.Zstart);
    model.initialize(input_data.vol);
    model.tilt_id = group_id;
//printf("ID:%d group_id:%d leader_id:%d N:%d \n",myid,group_id,leader_id,N);

//同步数据到所有线程
   
    //重构
    local_data.angN = input_data.prj.AngN;
    ComputeInitialModel(myid,&local_data,&input_data,&model,&opts);
    MPI_Barrier(MPI_COMM_WORLD); 
    vol_data_size = (long long)input_data.vol.X*input_data.vol.Y*input_data.vol.Z;
//	printf("vol_size:%lld \n",vol_data_size);
   //Comepute initial model(BPT) -Gather data 
    float *initial_model, *buffer; 
    int *displs,*recv_count;
	initial_model= (float *)malloc(sizeof(float)*vol_data_size);
    buffer= (float *)malloc(sizeof(float)*vol_data_size);
    memset(buffer,0,sizeof(float)*vol_data_size);
    recv_count = (int *)malloc(sizeof(int)*N);
    displs = (int  *)malloc(sizeof(int)*N);
    for(int i=0;i<N;i++)
    {
		Slice slc = divideSlice(i,N,input_data.vol.Z,input_data.vol.Zstart);
	    displs[i]=input_data.vol.X*input_data.vol.Y*(slc.z_start-input_data.vol.Zstart);
		recv_count[i]=input_data.vol.X*input_data.vol.Y*(slc.z_end-slc.z_start);
		printf("ID:%d I:%d disp:%d conunt:%d mysize:%d\n",myid,i,displs[i],recv_count[i],model.data_size);
    }
    MPI_Gatherv(model.output_data,model.data_size,MPI_FLOAT,buffer,recv_count,displs,MPI_FLOAT,0,tilt_comm);
    MPI_Reduce_BIG(buffer,initial_model, vol_data_size ,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD); 
    free(buffer);buffer=NULL;

	//Iteration
	MPI_Bcast_BIG(initial_model,vol_data_size ,MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD); 
	float *w = (float *)malloc(sizeof(float)*vol_data_size);
	float *z = (float *)malloc(sizeof(float)*vol_data_size);
	memcpy(w,initial_model,sizeof(float)*vol_data_size);
	clock_t start_time,end_time;
	start_time = clock();
	reconstructModel(myid,numprocs,w,z,&local_data,&input_data,initial_model,&model,&opts,recv_count,displs,group_id,tilt_comm,leader_id);
	end_time = clock();
	if(myid==0)printf("---------------------------------------------------Total time:%lf s ,%lf min\n",(double)(end_time-start_time)/CLOCKS_PER_SEC,(double)(end_time-start_time)/(CLOCKS_PER_SEC*60));
	
	if(myid==0) printf("Ready to write\n");
    if(myid==0)
    {
        if(updataAndWriteModel(&opts,&model,initial_model) < 0) {
            printf("***ERROR OCCURED WHEN WRITE MODEL.\n");
            exit(-1);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
	return 0;
}
