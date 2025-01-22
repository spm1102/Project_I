#include "tensor.h"

float **TENSOR2D_Get(float **input, int in_width, int in_height, float **tensor2d, int kernel_width, int kernel_height, int curr_row, int curr_col, int strides)
{
    // Check if the requested kernel fits within the input matrix
    if (curr_row + (kernel_width - 1) * strides + 1 > in_height ||
        curr_col + (kernel_height - 1) * strides + 1 > in_width)
    {
        printf("Kernel out of input matrix bounds!\n");
        return NULL;
    }

    for (int i = 0; i < kernel_width; ++i)
    {
        for (int j = 0; j < kernel_height; ++j)
        {
            int input_row = curr_row + i * strides;
            int input_col = curr_col + j * strides;
            tensor2d[i][j] = input[input_row][input_col];
        }
    }
    return tensor2d;
}

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