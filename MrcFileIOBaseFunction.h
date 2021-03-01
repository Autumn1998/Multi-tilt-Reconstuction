/*
 * MrcFileIOBaseFunction.h
 *
 *  Created on: Jan 26, 2021
 *      Author: liutong
 */

#ifndef MRCFILEIOBASEFUNCTION_H_
#define MRCFILEIOBASEFUNCTION_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sstream>
#include <string>

#define TEX_LINE_MAX 500
#define HEAD_SIZE 1024
#define INF 999999

#define PI_180 0.01745329252f
#ifndef PI
#define     PI  3.14159265358979323846
#endif
#define D2R(__ANGLE__) ((__ANGLE__) * PI_180) // PI / 180

#define SEEK_SET 0
#define SEEK_CUR 1
#define SEEK_END 2


#define MRC_MODE_BYTE          0
#define MRC_MODE_SHORT         1
#define MRC_MODE_FLOAT         2
#define MRC_MODE_COMPLEX_SHORT 3
#define MRC_MODE_COMPLEX_FLOAT 4
#define MRC_MODE_USHORT        6
#define MRC_MODE_RGB           16


#define MRC_LABEL_SIZE         80
#define MRC_NEXTRA             16
#define MRC_NLABELS            10
#define MRC_HEADER_SIZE        1024   /* Length of Header is 1024 Bytes. */
#define MRC_MAXCSIZE           3


using namespace std;

struct Slice
{
  int z_start;
  int z_end;
};

/*VolumeSize stores the parameter(not the data) of a volum data*/
typedef struct
{
  int Xstart;
  int Xend;
  int Ystart;
  int Yend;
  int Zstart;
  int Zend;

  int X;                       //X,Y,Z are the pixel size of the three dimesions in a volum data;
  int Y;
  int Z;                               //it also equals the thickness of a volum data;

} VolumeSize;

/*Proj stores the parameter(not the data) of a projection data*/
typedef struct
{
  int X;                       //X,Y,Z are the pixel size of the three dimesions in a volum data;
  int Y;
  int AngN;                               //it also equals the thickness of a volum data;
} ProjectionSize;

/*Pixel is the coordinate number of a 3d map*/
struct Pixel
{
	int    X;
	int    Y;
	int    Z;
};

/*computing proj by the coordinate of a 3D pixel*/
typedef struct
{
  int    x_min;//x coordinate of the proj
  int    y_min;//y coordinate of the proj

  double x_min_del;
  double y_min_del; //weight of the proj

} Weight;

#pragma pack(1)
struct MrcHeader
{
  int   nx;         /*  # of Columns                  */
  int   ny;         /*  # of Rows                     */
  int   nz;         /*  # of Sections.                */
  int   mode;       /*  given by #define MRC_MODE...  */

  int   nxstart;    /*  Starting point of sub image.  */
  int   nystart;
  int   nzstart;

  int   mx;         /* Grid size in x, y, and z       */
  int   my;
  int   mz;

  float   xlen;       /* length of x element in um.     */
  float   ylen;       /* get scale = xlen/nx ...        */
  float   zlen;

  float   alpha;      /* cell angles, ignore */
  float   beta;
  float   gamma;

  int   mapc;       /* map coloumn 1=x,2=y,3=z.       */
  int   mapr;       /* map row     1=x,2=y,3=z.       */
  int   maps;       /* map section 1=x,2=y,3=z.       */

  float   amin;
  float   amax;
  float   amean;

  short   ispg;       /* image type */
  short   nsymbt;     /* space group number */

  /* 64 bytes */

  int   next;
  short   creatid;  /* Creator id, hvem = 1000, DeltaVision = -16224 */

  char    blank[30];

  short   nint;
  short   nreal;

  short   sub;
  short   zfac;

  float   min2;
  float   max2;
  float   min3;
  float   max3;
  float   min4;
  float   max4;

  short   idtype;
  short   lens;
  short   nd1;     /* Devide by 100 to get float value. */
  short   nd2;
  short   vd1;
  short   vd2;
  float   tiltangles[6];  /* 0,1,2 = original:  3,4,5 = current */

  float   xorg;
  float   yorg;
  float   zorg;
  char    cmap[4];
  char    stamp[4];
  float   rms;

  int nlabl;
  char  labels[10][80];


} ;
#pragma pack()
/* END_CODE */
/******************************** Header functions or useful io functions **************************/

long get_file_size(FILE *fin);
int mrcReadhead(FILE *fin,  MrcHeader *head);

int mrcReadAllData(FILE *fin,float *mrc_data_all,long long st,long long ed,int mode);

int readCoef(VolumeSize *vol, double *x_coef, double *y_coef,int st, FILE *f_coef);

int cleanFile(char *file_addr);

int writeMRCHeader(FILE *fout,MrcHeader *head);

int writeMRCData(FILE *fout,MrcHeader *head, float *mrc_data_all);

#endif /* MRCFILEIOBASEFUNCTION_H_ */
