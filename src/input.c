#include "input.h"

#define MNIST_SIZE 28
#define RESNET_SIZE 224
#define BATCH_SIZE 1

void INPUT_read_file(const char *file_path, float ***buffer, int channel, int width, int height)
{
    FILE *file = fopen(file_path, "r");
    if (!file)
    {
        printf("Error: Cannot open file %s\n", file_path);
        exit(1);
    }

    for (int i = 0; i < channel; i++)
    {
        for (int j = 0; j < width; j++)
        {
            for (int k = 0; k < height; k++)
            {
                if (fscanf(file, "%f", &buffer[i][j][k]) != 1)
                {
                    printf("Error: Unexpected data format in file %s\n", file_path);
                    exit(1);
                }
            }
        }
    }

    fclose(file);
}

void read_file(const char *file_path, float ****buffer, int out_filter, int channel, int width, int height)
{
    FILE *file = fopen(file_path, "r");
    if (file == NULL)
    {
        printf("Error: Cannot open file %s\n", file_path);
        exit(1);
    }

    for (int i = 0; i < out_filter; i++)
    {
        for (int j = 0; j < channel; j++)
        {
            for (int k = 0; k < width; k++)
            {
                for (int z = 0; z < height; z++)
                {
                    if (fscanf(file, "%f", &buffer[i][j][k][z]) != 1)
                    {
                        printf("Error: Unexpected data format in file %s\n", file_path);
                        exit(1);
                    }
                }
            }
        }
    }
    fclose(file);
}

void FC_readData(const char *file_path, float **buffer, int width, int height)
{
    FILE *file = fopen(file_path, "r");
    if (file == NULL)
    {
        printf("Error: Cannot open file %s\n", file_path);
        exit(1);
    }
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            if (fscanf(file, "%f", &buffer[i][j]) != 1)
            {
                printf("Error: Unexpected data format in file %s\n", file_path);
                exit(1);
            }
        }
    }
}

void batchNorm_readData(const char *file_path, float *buffer, int size)
{
    FILE *file = fopen(file_path, "r");
    if (!file)
    {
        printf("Error: Cannot open file %s \n", file_path);
    }
    for (int i = 0; i < size; i++)
    {
        if (fscanf(file, "%f", &buffer[i]) != 1)
        {
            printf("Error: Unexpected data format in file %s\n", file_path);
            exit(1);
        }
    }
}

void resize_image(float ***input, float ***output)
{
    float scale = (float)(MNIST_SIZE - 1) / (RESNET_SIZE - 1);

    // Xử lý cho batch_size = 1
    for (int b = 0; b < BATCH_SIZE; b++)
    {
        for (int i = 0; i < RESNET_SIZE; i++)
        {
            for (int j = 0; j < RESNET_SIZE; j++)
            {
                float src_x = j * scale;
                float src_y = i * scale;
                int x0 = (int)src_x;
                int x1 = (x0 + 1 < MNIST_SIZE) ? x0 + 1 : x0;
                int y0 = (int)src_y;
                int y1 = (y0 + 1 < MNIST_SIZE) ? y0 + 1 : y0;

                float dx = src_x - x0;
                float dy = src_y - y0;

                float value = (1 - dx) * (1 - dy) * input[b][y0][x0] +
                            dx * (1 - dy) * input[b][y0][x1] +
                            (1 - dx) * dy * input[b][y1][x0] +
                            dx * dy * input[b][y1][x1];

                output[b][i][j] = value;
            }
        }
    }
}


