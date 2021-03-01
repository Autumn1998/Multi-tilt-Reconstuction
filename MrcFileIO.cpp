#include "MrcFileIO.h"

void OutputModel::headerInitialize()
{
  output_header.nx=0;
  output_header.ny=0;
  output_header.nz=0;

  output_header.mode=MRC_MODE_FLOAT;

  output_header.nxstart=0;
  output_header.nystart=0;
  output_header.nzstart=0;

  output_header.mx=1;
  output_header.my=1;
  output_header.mz=1;

  output_header.xlen=1;
  output_header.ylen=1;
  output_header.zlen=1;

  output_header.alpha=90;
  output_header.beta=90;
  output_header.gamma=90;

  output_header.mapc=1;
  output_header.mapr=2;
  output_header.maps=3;

  output_header.amin=0;
  output_header.amax=255;
  output_header.amean=128;

  output_header.ispg=0;
  output_header.nsymbt=0;

  output_header.next=0;

  output_header.creatid=1000;
  output_header.cmap[0]='M';
  output_header.cmap[1]='A';
  output_header.cmap[2]='P';

  output_header.stamp[0]='D';
}

void OutputModel::headerSetValue()
{
  output_header.nx=vol.X;
  output_header.ny=vol.Y;
  output_header.nz=vol.Z;

  output_header.nxstart=vol.Xstart;
  output_header.nystart=vol.Ystart;
  output_header.nzstart=vol.Zstart;

  output_header.mx=vol.X;
  output_header.my=vol.Y;
  output_header.mz=vol.Z;
}

void OutputModel::updateHeader(float *data)
{
  long double sum=0,amin,amax,amean;
  int prj_size=output_header.nx*output_header.ny,i,j;
  printf(">>>>updating head(FLOAT)...\n");
  amax = amin = output_data[0];
  for(j = 0;j<output_header.nz;j++)
  {
    amean = 0;
    for(i = 0;i<prj_size;i++)
    {
      int tmp_index = i+j*prj_size;
      if(data[tmp_index]>amax) amax = data[tmp_index];
      if(data[tmp_index]<amin) amin = data[tmp_index];
      amean+=data[tmp_index];
    }
    amean/=prj_size;
    sum += amean;
  }
  amean = sum/output_header.nz;
  printf("%Lf %Lf %Lf\n",amax,amin,amean);
  output_header.amin=(float)amin;
  output_header.amax=(float)amax;
  output_header.amean=(float)amean;
  printf(">>>>output_header.amin is %f, output_header.amax is %f, output_header.amean is %f\n",output_header.amin, output_header.amax, output_header.amean);
}

void OutputModel::printHeader()
{
  printf("%d %d %d\n",output_header.nx,output_header.ny,output_header.nz);
  printf("xs:%d xe:%d x:%d\n",vol.Xstart, vol.Xend, vol.X);
  printf("ys:%d ye:%d y:%d\n",vol.Ystart, vol.Yend, vol.Y);
  printf("zs:%d ze:%d z:%d\n",vol.Zstart, vol.Zend, vol.Z);
}

void OutputModel::initialize(VolumeSize v)
{
  headerInitialize();
  vol = v;
  data_size = vol.X*vol.Y*(slc.z_end-slc.z_start);
  output_data = (float *)malloc(sizeof(float)*data_size);
  headerSetValue();
}

int readMrcHeader(Options *opt, InputData *input_data)
{
      int i = input_data->group_id;
      FILE *in_file = fopen(opt->mrc_dir[i].c_str(),"r");
      if(!in_file){
        printf("Can not open in_file\n");
        return false;
      }
      mrcReadhead(in_file,&input_data->mrc_header);
      //printf("xsize:%d ysize:%d zsize:%d\n",oldx,oldy,oldz);
      //printf("input_data->mrc_headers[i].nx:%d input_data->mrc_headers[i].ny:%d zsize:%d\n",input_data->mrc_header.nx, input_data->mrc_header.ny,input_data->mrc_header.nz);
      fclose(in_file);
      
    input_data->prj.X = input_data->mrc_header.nx;
    input_data->prj.Y = input_data->mrc_header.nx;
    input_data->prj.AngN = input_data->mrc_header.nz;

    return true;
}

int readRemainMRCData(Options *opt, InputData *input_data,float *buffer)
{
    int i=input_data->group_id;
    int angN = input_data->prj.AngN;
  long long st=0,ed,size_one_read;
	FILE *mrc_file=fopen(opt->mrc_dir[i].c_str(),"r");
    if(!mrc_file){
      printf("Can not open mrc_file\n");
      return false;
    }
    size_one_read=input_data->prj.X*input_data->prj.Y*angN;
    ed=st+size_one_read;
    mrcReadAllData(mrc_file,buffer,st,ed,input_data->mrc_header.mode);
    fclose(mrc_file);
  return true;
}

int readTXBRData(Options *opt, InputData *input_data,DataToUse *local_data)
{
/*  local_data->x_coef = (double *)malloc(sizeof(double *)*input_data->prj.AngN*10);
  memset(input_data->x_coef, 0 , sizeof(double *)*input_data->prj.AngN*10);
  local_data->y_coef = (double *)malloc(sizeof(double *)*input_data->prj.AngN*10);
  memset(input_data->y_coef, 0 , sizeof(double *)*input_data->prj.AngN*10);
*/
      int i = input_data->group_id,pos=0;
	FILE *angle_file = fopen(opt->txbr_dir[i].c_str(),"r");
    if(!angle_file){
      printf("Can not open angle_file\n");
      return false;
    }
    readCoef(&input_data->vol,local_data->x_coef, local_data->y_coef,pos, angle_file);
    fclose(angle_file);
  return true;
}

int readMrcAndTXBR(Options *opt, InputData *input_data,DataToUse *local_data)
{
    //printf("<<<<<<<<<<<<<  Reading MRC file...\n");
    if(readRemainMRCData(opt, input_data, local_data->mrc_data) < 0) {
        printf("***WRONG AT READ REMAIN MRC DATA.\n");
        return -1;
    }
    //printf(">>>>>>>>>>>>>  Finished.\n");

    //printf("<<<<<<<<<<<<<  Reading TXBR file...\n");
    if(readTXBRData(opt, input_data,local_data) < 0) {
        printf("***WRONG AT READ REMAIN MRC DATA.\n");
        return -1;
    }
    //printf(">>>>>>>>>>>>>  Finished.\n");
    return 1;
}

int writeModel(Options *opts, OutputModel *model_info,float *data)
{
  cleanFile(opts->out_dir);
	FILE *out_file = fopen(opts->out_dir,"r+");
	if(!out_file){
		printf("Can not open out_file!\n");
		return -1;
	}
	writeMRCHeader(out_file,&model_info->output_header);
	writeMRCData(out_file,&model_info->output_header,data);
	fclose(out_file);
	return 1;
}

int updataAndWriteModel(Options *opts, OutputModel *model_info, float *data)
{
	printf(">>>>>>>>>>>>>  OUT head:%d %d %d 0\n",model_info->output_header.nx,model_info->output_header.ny,model_info->output_header.nz);
	model_info->updateHeader(data);
  writeModel(opts,model_info,data);
  printf(">>>>>>>>>>>>>  Write finished! :%s\n",opts->out_dir);
  return 1;
}
