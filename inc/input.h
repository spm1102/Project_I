#ifndef INPUT_H
#define INPUT_H
#include<stdio.h>
#include<stdlib.h>

#include"conv2d.h"

void INPUT_read_file(const char *file_path, float ***buffer, int channel, int width, int height);
void read_file(const char *file_path, float ****buffer, int out_filter, int channel, int width, int height);

void FC_readData(const char *file_path, float** buffer, int width, int height);
void batchNorm_readData(const char *file_path, float* buffer, int size);
void resize_image(float*** input, float*** output);
#endif