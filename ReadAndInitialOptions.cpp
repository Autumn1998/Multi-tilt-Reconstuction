/*
 * ReadAndInitialOptions.cpp
 *
 *  Created on: Jan 26, 2021
 *      Author: liutong
 */

#include "ReadAndInitialOptions.h"

#include "ReadAndInitialOptions.h"
#include <cuda_runtime.h>
#define MAX_TILT 16
#define G_MEM 1073741824

using namespace std;

int checkGPUMem(int device_id)
{
    cudaSetDevice(device_id);
	size_t avail;
	size_t total;
    cudaMemGetInfo(&avail,&total);
	printf("avail:%d total: %d\n ",avail,total);
    return true;
}

void getGPUInfo(int device_id)
{
    int driver_version(0), runtime_version(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_id);
    if (device_id == 0)
        if (deviceProp.minor = 9999 && deviceProp.major == 9999)
            printf("\n");
    printf("\nDevice%d:\"%s\"\n", device_id, deviceProp.name);
    cudaDriverGetVersion(&driver_version);
    printf("CUDA驱动版本:                                   %d.%d\n", driver_version / 1000, (driver_version % 1000) / 10);
    cudaRuntimeGetVersion(&runtime_version);
    printf("CUDA运行时版本:                                 %d.%d\n", runtime_version / 1000, (runtime_version % 1000) / 10);
    printf("设备计算能力:                                   %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("Total amount of Global Memory:                  %u bytes\n", deviceProp.totalGlobalMem);
    printf("Number of SMs:                                  %d\n", deviceProp.multiProcessorCount);
    printf("Total amount of Constant Memory:                %u bytes\n", deviceProp.totalConstMem);
    printf("Total amount of Shared Memory per block:        %u bytes\n", deviceProp.sharedMemPerBlock);
    printf("Total number of registers available per block:  %d\n", deviceProp.regsPerBlock);
    printf("Warp size:                                      %d\n", deviceProp.warpSize);
    printf("Maximum number of threads per SM:               %d\n", deviceProp.maxThreadsPerMultiProcessor);
    printf("Maximum number of threads per block:            %d\n", deviceProp.maxThreadsPerBlock);
    printf("Maximum size of each dimension of a block:      %d x %d x %d\n", deviceProp.maxThreadsDim[0],
        deviceProp.maxThreadsDim[1],
        deviceProp.maxThreadsDim[2]);
    printf("Maximum size of each dimension of a grid:       %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    printf("Maximum memory pitch:                           %u bytes\n", deviceProp.memPitch);
    printf("Texture alignmemt:                              %u bytes\n", deviceProp.texturePitchAlignment);
    printf("Clock rate:                                     %.2f GHz\n", deviceProp.clockRate * 1e-6f);
    printf("Memory Clock rate:                              %.0f MHz\n", deviceProp.memoryClockRate * 1e-3f);
    printf("Memory Bus Width:                               %d-bit\n", deviceProp.memoryBusWidth);
}

void helpInfo()
{
    printf("[-i INPUT FILENAME]\n");
    printf("[-i INPUT FILENAME]\n");
	printf("    MRC file for reconstruction\n");
	printf("[-o OUTPUT FILENAME]\n");
	printf("    MRC filename for result\n");
	printf("[-p PARAMETER FILENAME]\n");
	printf("    Projection parameters\n");
	printf("([-n INITIAL RECONSTRUCTION])\n");
	printf("    MRC file as initial model (reconstruction) for iteration methods (optinal)\n");
	printf("[-g O,P,Z,T]\n");
	printf("    Geometry information: offset,pitch_angle,zshift,thickness\n");
	printf("[-m METHOD,I,R]\n");
	printf("    BackProjection: BPT\n");
	printf("    SART: SART,iteration_number,relax_parameter\n");
    printf("    SIRT: SIRT,iteration_number,relax_parameter\n");
	printf("    Reprojection: RP\n");
	printf("[-h]");
	printf("    Help Information\n");
    printf("EXAMPLES:\n");
}


inline void printOptions(Options *opts)
{
    printf("---------------------------------OUTPUT OPTIONS--------------------------------------------\n");
    printf("cuda block x =%d\n",opts->block_x);
    printf("cuda block y =%d\n",opts->block_y);
    printf("cuda block z =%d\n",opts->block_z);
    printf("out address =%s\n",opts->out_dir);
    printf("iteration times =%d\n",opts->iter_times);
    printf("sirt iteration step =%f\n",opts->sirt_step);
    printf("mace iteration step =%f\n",opts->mace_step);
    printf("tilt number =%d\n",opts->number_tilt);
    printf("slice number =%d\n",opts->slice_num);
    for(int i=0;i<opts->number_tilt;i++)
    {
        printf("%d: mrcfile address =%s",i,opts->mrc_dir[i].data());
        printf("    txbr address =%s\n",opts->txbr_dir[i].data());
    }
    printf("-------------------------------------------------------------------------------------------\n");
}

void praseInputAddr(char *args, Options *opts)
{
    stringstream iss(args);
    string tmp;
    getline(iss, tmp, ',');
    opts->number_tilt = atoi(tmp.c_str());
    opts->mrc_dir = new string[opts->number_tilt];
    opts->txbr_dir = new string[opts->number_tilt];

    for(int i=0;i<opts->number_tilt;i++)
    {
        getline(iss, tmp, ',');
        opts->mrc_dir[i] = tmp;
    }

    for(int i=0;i<opts->number_tilt;i++)
    {
        if(i<opts->number_tilt-1) getline(iss, tmp, ',');
        else getline(iss, tmp);
        opts->txbr_dir[i] = tmp;
    }
}

int readOptions(int argc,char **argv,Options* const &opts)
{
    static struct option longopts[] ={
        { "help",                      no_argument,            NULL,              'h' },
        { "input mrc file number",     required_argument,      NULL,              'i' },
        { "output mrc file",           required_argument,      NULL,              'o' },
        { "cuda block size -x",        required_argument,      NULL,              'x' },
        { "cuda block size -y",        required_argument,      NULL,              'y' },
        { "cuda block size -z",        required_argument,      NULL,              'z' },
        { "iteration times",           required_argument,      NULL,              'n' },
        { "sirt iteration step",       required_argument,      NULL,              's' },
        { "mace iteration step",       required_argument,      NULL,              'm' },
        { NULL,                        0,                      NULL,               0  }
     };

    char ch;
    while((ch = getopt_long(argc, argv, "hi:o:x:y:z:n:s:m:", longopts, NULL)) != -1)
    {
        switch (ch)
        {
            case '?':  printf("Invalid option '%s'.", argv[optind-1]);
                return -1;
	    	case ':':  printf("Missing option argument for '%s'.", argv[optind-1]);
                return -1;
            case 'h':  helpInfo();
                return 0;

            case 'x':  opts->block_x = atoi(optarg); break;
            case 'y':  opts->block_y = atoi(optarg); break;
            case 'z':  opts->block_z = atoi(optarg); break;

            case 'n':  opts->iter_times = atoi(optarg); break;
            case 's':  opts->sirt_step = atof(optarg); break;
            case 'm':  opts->mace_step = atof(optarg); break;
            case 'i':
            {
                praseInputAddr(optarg,opts);
                break;
            }
            case 'o':  opts->out_dir = optarg;  break;
            default: break;
        }
    }
    //printOptions(opts);
    return 1;
}
