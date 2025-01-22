#ifndef CONV2D
#define CONV2D

#include<math.h>

#include "kernel.h"
#include "matrix.h"
#include "tensor.h"

#define EPSILON 1e-5

float ***Conv2D(float ***input, int in_width, int in_height, int in_filters, int out_filters, float ****kernel, int kernel_width, int kernel_height, int padding, int strides, int bias);
float ***Padding(float ***input, int in_width, int in_height, int in_filters, int padding);
float ***MAX_Pooling(float ***input, int in_width, int in_height, int in_filters, int kernel_size, int strides, int padding);
float ***AVG_Pooling(float ***input, int in_width, int in_height, int in_filters, int kernel_size, int strides, int padding);

float exp_sum(float *input, int length);
float* Softmax(float *input, int length);
float ***ReLU(float ***input, int in_width, int in_height, int in_filters);
float *Flatten(float*** input, int in_width, int in_height, int in_filters);
float ***ADD_SKIP_CONNECTION(float*** input, float*** output, int width, int height, int channels);
void batchnorm_4d(float ***input, int height, int width, int num_channels, float *gamma, float *beta, float epsilon);
void BatchNorm_free(float *gamma, float *beta);
#endif