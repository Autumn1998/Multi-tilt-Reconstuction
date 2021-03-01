/*
 * MrcFileIO.h
 *
 *  Created on: Jan 26, 2021
 *      Author: liutong
 */

#ifndef MRCFILEIO_H_
#define MRCFILEIO_H_
#include "MrcFileIOBaseFunction.h"
#include "ReadAndInitialOptions.h"

struct InputData
{
    float *mrc_data;
    double *x_coef, *y_coef;
    MrcHeader mrc_header;
    ProjectionSize prj;
    VolumeSize vol;
    int group_id;
    int thread_id;
    InputData()
    {
      vol.Xstart = INF; vol.Xend = -INF;
      vol.Ystart = INF; vol.Yend = -INF;
      vol.Zstart = INF; vol.Zend = -INF;
      vol.Z = 91;
    }
};

struct OutputModel
{
    MrcHeader output_header;
    float *output_data;
    VolumeSize vol;
    Slice slc;
    long long data_size;
    int tilt_id;

    void initialize(VolumeSize v);
    void headerInitialize();
    void updateHeader(float *data);
    void printHeader();
    void headerSetValue();
};

struct DataToUse
{
  int tilt_id;
  int angN;
  float *mrc_data;
  double *x_coef, *y_coef;
};

int readMrcHeader(Options *opt, InputData *input_data);

int readRemainMRCData(Options *opt, InputData *input_data,float *buffer,int i,int angN);

int readTXBRData(Options *opt, InputData *input_data);

int readMrcAndTXBR(Options *opt, InputData *input_data, DataToUse *local_data);

int writeModel(Options *opts, OutputModel *model_info, float *data);

int updataAndWriteModel(Options *opts, OutputModel *model_info,float *data);


#endif /* MRCFILEIO_H_ */
