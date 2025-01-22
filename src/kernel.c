#include "kernel.h"

float ****KERNEL_RandomCreate(float ****kernel, int out_filter, int channel, int width, int height)
{
    for (int i = 0; i < out_filter; i++)
    {
        for (int j = 0; j < channel; j++)
        {
            for (int k = 0; k < width; k++)
            {
                for (int z = 0; z < height; z++)
                {
                    kernel[i][j][k][z] = (float)rand() / RAND_MAX;
                }
            }
        }
    }
    return kernel;
}