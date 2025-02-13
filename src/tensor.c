#include "tensor.h"

float ***TENSOR3D_Create(int channel, int width, int height)
{
    float ***tensor3D = (float ***)malloc(channel * sizeof(float **));
    if (!tensor3D)
    {
        return NULL;
    }
    for (int i = 0; i < channel; i++)
    {
        tensor3D[i] = (float **)malloc(width * sizeof(float *));
        if (!tensor3D[i])
            return NULL;
        for (int j = 0; j < width; j++)
        {
            tensor3D[i][j] = (float *)malloc(height * sizeof(float));
            if (!tensor3D[i][j])
                return NULL;
        }
    }

    return tensor3D;
}

void TENSOR3D_Free(float ***tensor3D, int channel, int width, int height)
{
    for (int i = 0; i < channel; i++)
    {
        for (int j = 0; j < width; j++)
        {
            free(tensor3D[i][j]);
        }
        free(tensor3D[i]);
    }
    free(tensor3D);
}

float ****TENSOR4D_Create(int out_channel, int in_channel, int width, int height)
{
    float ****tensor4D = (float ****)malloc(out_channel * sizeof(float ***));
    if (!tensor4D)
    {
        return NULL;
    }
    for (int i = 0; i < out_channel; i++)
    {
        tensor4D[i] = (float ***)malloc(in_channel * sizeof(float **));
        if (!tensor4D[i])
            return NULL;
        for (int j = 0; j < in_channel; j++)
        {
            tensor4D[i][j] = (float **)malloc(width * sizeof(float *));
            if (!tensor4D[i][j])
                return NULL;
            for (int k = 0; k < width; k++)
            {
                tensor4D[i][j][k] = (float *)malloc(height * sizeof(float));
                if (!tensor4D[i][j][k])
                    return NULL;
            }
        }
    }
    return tensor4D;
}


void TENSOR4D_Free(float**** tensor4D, int out_channel, int in_channel, int width, int height) {
    if (tensor4D == NULL) return; 
    
    for (int o = 0; o < out_channel; o++) {
        if (tensor4D[o] != NULL) {
            for (int i = 0; i < in_channel; i++) {
                if (tensor4D[o][i] != NULL) {
                    for (int w = 0; w < width; w++) {
                        free(tensor4D[o][i][w]); 
                    }
                    free(tensor4D[o][i]); 
                }
            }
            free(tensor4D[o]);
        }
    }
    free(tensor4D); 
}