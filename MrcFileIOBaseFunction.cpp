#include "MrcFileIOBaseFunction.h"

using namespace std;

long get_file_size(FILE *fin)
{
	fseek(fin,0,SEEK_END);
	return ftell(fin);
}

int mrcReadhead(FILE *fin,  MrcHeader *head)
{
  if(ftello64(fin)!=0)rewind(fin);
  fread(head,sizeof(char),HEAD_SIZE,fin);
  return true;
}

int mrcReadAllData(FILE *fin,float *mrc_data_all,long long st,long long ed,int mode)
{
	int headsize = get_file_size(fin) - (ed-st) * sizeof(short);
    //compute the offset

	unsigned char buf_byte;
	short buf_short;
	short buf_ushort;
    float buf_float;

	fseek(fin,headsize,SEEK_SET);

	switch(mode)
	{
		case MRC_MODE_BYTE:
			printf("MRC_MODE_TYPE = BYTE\n");
			for(long i=st;i<ed;i++)
			{
				fread(&buf_byte,sizeof(char),1,fin);
				mrc_data_all[i] = (float)buf_byte;
			}
			break;
		case MRC_MODE_SHORT:
			printf("MRC_MODE_TYPE = SHORT\n");
			for(long i=st;i<ed;i++)
			{
				fread(&buf_short,sizeof(short),1,fin);
				mrc_data_all[i] = (float)buf_short;
			}
			break;
		case MRC_MODE_USHORT:
			printf("MRC_MODE_TYPE = USHORT\n");
			for(long i=st;i<ed;i++)
			{
				fread(&buf_ushort,sizeof(char),1,fin);
				mrc_data_all[i] = (float)buf_ushort;
			}
			break;
		case MRC_MODE_FLOAT:
			printf("MRC_MODE_TYPE = FLOAT\n");
			for(long i=st;i<ed;i++)
			{
				fread(&buf_float,sizeof(char),1,fin);
				mrc_data_all[i] = (float)buf_float;
                if(mrc_data_all[i] != 0 ) {printf("--------------%d:%f  %f\n",i,mrc_data_all[i],buf_float);break;}
			}
			break;
		default:
			printf("Error with Function 'mrc_read_all'!File type unknown!");
			break;
	}
	return 0;
}

int readCoef(VolumeSize *vol, double *x_coef, double *y_coef,int st, FILE *f_coef)
{
	char *line_buff = (char *)malloc(TEX_LINE_MAX);
	char *tmp;
	int i=st,j=st,ang_num=0;
	while(fgets(line_buff,TEX_LINE_MAX,f_coef)!=NULL)
	{
		if(line_buff[0] == 'l') ang_num++;
		else if(line_buff[0] == 'x' && line_buff[1] == '-' && line_buff[2] != '>')
		{
			tmp = strtok(line_buff," ");
			while(tmp!=NULL)
			{
				tmp = strtok(NULL," ");
				if(tmp!=NULL)
				{
					x_coef[i++]=strtod(tmp,NULL);
					//printf("x_coef[%d]:%lf\n",i-1,x_coef[i-1]);
				}
			}
		}else if(line_buff[0] == 'y' && line_buff[1] == '-' && line_buff[2] != '>')
		{
			tmp = strtok(line_buff," ");
			while(tmp!=NULL)
			{
				tmp = strtok(NULL," ");
				if(tmp!=NULL)
				{
					y_coef[j++]=strtod(tmp,NULL);
					//printf("y_codf[%d]:%lf\n",j-1,y_coef[j-1]);
				}
			}
		}else if(line_buff[0] == 'x' && line_buff[1] == '-' && line_buff[2] == '>')
		{
			line_buff[2] = ':';
			tmp = strtok(line_buff,":");tmp = strtok(NULL,":");
			vol->Xstart = min(vol->Xstart,atoi(tmp) );
			tmp = strtok(NULL,":");tmp = strtok(NULL,":");
			vol->Xend = max(vol->Xend,atoi(tmp) );
		}else if(line_buff[0] == 'y' && line_buff[1] == '-' && line_buff[2] == '>')
		{
			line_buff[2] = ':';
			tmp = strtok(line_buff,":");tmp = strtok(NULL,":");
			vol->Ystart = min(vol->Ystart,atoi(tmp) );
			tmp = strtok(NULL,":");tmp = strtok(NULL,":");
			vol->Yend = max(vol->Yend,atoi(tmp) );

		}else if(line_buff[0] == 'z' && line_buff[1] == '-' && line_buff[2] == '>')
		{
			line_buff[2] = ':';
			tmp = strtok(line_buff,":");tmp = strtok(NULL,":");
			vol->Zstart = min(vol->Zstart,atoi(tmp) );
			tmp = strtok(NULL,":");	tmp = strtok(NULL,":");
			vol->Zend = max(vol->Zend,atoi(tmp) );
		}
	}
	//printf("Angle_num:%d\n",ang_num);
    vol->X = vol->Xend - vol->Xstart;
	vol->Y = vol->Yend - vol->Ystart;
	vol->Z = vol->Zend - vol->Zstart;
	if(i==j) return i;
    else return -1;
}

int cleanFile(char *file_addr)
{
	FILE * file = fopen(file_addr,"w");
	if(!file) printf("open file failed \n");
	fclose(file);
	return true;
}

int writeMRCHeader(FILE *fout,MrcHeader *head)
{
	if(ftello64(fout)!=0)rewind(fout);
	//printf("To be written:%d %d %d \n",head->nx,head->ny,head->nz);
	fwrite(head,sizeof(char),HEAD_SIZE,fout);
	return true;
}

int writeMRCData(FILE *fout,MrcHeader *head, float *mrc_data_all)
{
    MrcHeader *h=(MrcHeader *)malloc(sizeof(MrcHeader));
	mrcReadhead(fout,h);
	fseek(fout, HEAD_SIZE, SEEK_SET );
	//printf("Mode:%d Write all:%d %d %d \n",head->mode,head->nx,head->ny,head->nz);
    fwrite(mrc_data_all,sizeof(float),head->nx*head->ny*head->nz,fout);
	return true;
}
