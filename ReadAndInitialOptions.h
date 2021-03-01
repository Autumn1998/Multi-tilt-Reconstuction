/*
 * ReadAndInitialOptions.h
 *
 *  Created on: Jan 26, 2021
 *      Author: liutong
 */

#ifndef READANDINITIALOPTIONS_H_
#define READANDINITIALOPTIONS_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <sstream>
#include <iostream>
#define MAX_TILT 16
#define G_MEM 1073741824

using namespace std;

int checkGPUMem(int device_id);
void getGPUInfo(int device_id);

struct Options
{
    string *mrc_dir,*txbr_dir;
    int number_tilt,block_x,block_y,block_z,iter_times;
    float sirt_step,mace_step;
    char *out_dir;
    int slice_num;

    Options()
    {
        block_x = 8;
        block_y = 8;
        block_z = 8;
        iter_times = 5;
        sirt_step = 0.5;
        slice_num = 0;
		mace_step = 0.7;
    }
};

void helpInfo();

void printOptions(Options *opts);

void praseInputAddr(char *args, Options *opts);
int readOptions(int argc,char **argv,Options* const &opts);

#endif /* READANDINITIALOPTIONS_H_ */
