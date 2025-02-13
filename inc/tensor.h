#ifndef TENSOR
#define TENSOR
#include<stdio.h>
#include<stdlib.h>

float ***TENSOR3D_Create(int channel, int width, int height);
void TENSOR3D_Free(float ***tensor3D, int channel, int width, int height);
float ****TENSOR4D_Create(int out_channel, int in_channel, int width, int height);
void TENSOR4D_Free(float**** tensor4D, int out_channel, int in_channel, int width, int height);

#endif