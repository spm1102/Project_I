#include "conv2d.h"

float ***Conv2D(float ***input, int in_width, int in_height, int in_filters, int out_filters,
                 float ****kernel, int kernel_width, int kernel_height, int padding, int strides, int bias)
{
    const int out_width = (in_width - kernel_width + 2 * padding) / strides + 1;
    const int out_height = (in_height - kernel_height + 2 * padding) / strides + 1;

    float ***output = TENSOR3D_Create(out_filters, out_width, out_height);
    float ***padded_input = NULL;
    if (padding)
    {
        padded_input = Padding(input, in_width, in_height, in_filters, padding);
        in_width = in_width + 2 * padding;
        in_height = in_height + 2 * padding;
    }
    else
    {
        padded_input = input;
    }

    for (int out_channel = 0; out_channel < out_filters; out_channel++)
    {
        for (int out_row = 0; out_row < out_width; out_row++)
        {
            for (int out_col = 0; out_col < out_height; out_col++)
            {
                float channel_output = 0.0;
                for (int in_channel = 0; in_channel < in_filters; in_channel++)
                {
                    for (int i = 0; i < kernel_width; i++)
                    {
                        int input_row = out_row * strides + i;
                        for (int j = 0; j < kernel_height; j++)
                        {
                            int input_col = out_col * strides + j;
                            // check out of range
                            if (input_row >= 0 && input_row < in_height && input_col >= 0 && input_col < in_width)
                            {
                                channel_output += padded_input[in_channel][input_row][input_col] * kernel[out_channel][in_channel][i][j];
                            }
                            else
                            {
                                break;
                            }
                        }
                    }
                }
                // Add bias if enabled (simplified)
                output[out_channel][out_row][out_col] = channel_output + (bias ? bias : 0);
            }
        }
    }
    if (padding)
        TENSOR3D_Free(padded_input, in_filters, in_width, in_height);
    return output;
}

float ***Padding(float ***input, int in_width, int in_height, int in_filters, int padding)
{
    int padded_width = in_width + 2 * padding;
    int padded_height = in_height + 2 * padding;

    // Create new tensor with all value = 0
    float ***padded_input = TENSOR3D_Create(in_filters, padded_width, padded_height);
    for (int i = 0; i < in_filters; i++)
    {
        for (int j = 0; j < padded_width; j++)
        {
            for (int k = 0; k < padded_height; k++)
            {
                padded_input[i][j][k] = 0;
            }
        }
    }

    // Assign value from input to padded_input
    for (int i = 0; i < in_filters; i++)
    {
        for (int j = 0; j < in_width; j++)
        {
            for (int k = 0; k < in_height; k++)
            {
                padded_input[i][j + padding][k + padding] = input[i][j][k];
            }
        }
    }
    return padded_input;
}

float ***MAX_Pooling(float ***input, int in_width, int in_height, int in_filters, int kernel_size, int strides, int padding)
{

    const int out_width = (in_width - kernel_size + 2 * padding) / strides + 1;
    const int out_height = (in_height - kernel_size + 2 * padding) / strides + 1;
    float ***output = TENSOR3D_Create(in_filters, out_width, out_height);

    if (padding)
    {
        input = Padding(input, in_width, in_height, in_filters, padding);
        in_width = in_width + 2 * padding;
        in_height = in_height + 2 * padding;
    }

    for (int out_channel = 0; out_channel < in_filters; out_channel++)
    {
        for (int out_row = 0; out_row < out_width; out_row++)
        {
            for (int out_col = 0; out_col < out_height; out_col++)
            {
                float max_value = -__DBL_MAX__;
                for (int i = 0; i < kernel_size; i++)
                {
                    for (int j = 0; j < kernel_size; j++)
                    {
                        int row = out_row * strides + i;
                        int col = out_col * strides + j;

                        // Bounds check to avoid invalid memory access
                        if (row >= 0 && row < in_width && col >= 0 && col < in_height)
                        {
                            if (input[out_channel][row][col] > max_value)
                            {
                                max_value = input[out_channel][row][col];
                            }
                        }
                        else
                        {
                            break;
                        }
                    }
                }
                output[out_channel][out_row][out_col] = max_value;
            }
        }
    }
    return output;
}

float ***AVG_Pooling(float ***input, int in_width, int in_height, int in_filters, int kernel_size, int strides, int padding)
{

    const int out_width = (in_width - kernel_size + 2 * padding) / strides + 1;
    const int out_height = (in_height - kernel_size + 2 * padding) / strides + 1;
    float ***output = TENSOR3D_Create(in_filters, out_width, out_height);

    if (padding)
    {
        input = Padding(input, in_width, in_height, in_filters, padding);
        in_width = in_width + 2 * padding;
        in_height = in_height + 2 * padding;
    }

    for (int out_channel = 0; out_channel < in_filters; out_channel++)
    {
        for (int out_row = 0; out_row < out_width; out_row++)
        {
            for (int out_col = 0; out_col < out_height; out_col++)
            {
                float avg_value = 0;
                for (int i = 0; i < kernel_size; i++)
                {
                    for (int j = 0; j < kernel_size; j++)
                    {
                        int row = out_row * strides + i;
                        int col = out_col * strides + j;

                        // Bounds check to avoid invalid memory access
                        if (row >= 0 && row < in_width && col >= 0 && col < in_height)
                        {
                            avg_value += input[out_channel][row][col];
                        }
                        else
                        {
                            break;
                        }
                    }
                }
                output[out_channel][out_row][out_col] = avg_value / (float)(kernel_size * kernel_size);
            }
        }
    }
    return output;
}

float exp_sum(float *input, int length)
{
    float exp_sum = 0;
    float max_val = -__DBL_MAX__;
    
    // Tìm max
    for (int i = 0; i < length; i++)
    {
        if (input[i] > max_val)
            max_val = input[i];
    }

    // Tính tổng exp
    for (int i = 0; i < length; i++)
    {
        exp_sum += exp(input[i] - max_val);
    }
    
    return exp_sum;
}

float* Softmax(float *input, int length)
{
    float sum_exp = exp_sum(input, length);
    float max_val = -__DBL_MAX__;
    
    // Tìm max
    for (int i = 0; i < length; i++)
    {
        if (input[i] > max_val)
            max_val = input[i];
    }

    // Áp dụng softmax trực tiếp vào input
    for (int i = 0; i < length; i++)
    {
        input[i] = exp(input[i] - max_val) / sum_exp;
    }
    
    return input;
}

float ***ReLU(float ***input, int in_width, int in_height, int in_filters)
{
    for (int i = 0; i < in_filters; i++)
    {
        for (int j = 0; j < in_width; j++)
        {
            for (int k = 0; k < in_height; k++)
            {
                input[i][j][k] = (input[i][j][k] > 0) ? input[i][j][k] : 0.0;
            }
        }
    }
    return input;
}

float *Flatten(float ***input, int in_width, int in_height, int in_filters)
{
    float *result = (float *)malloc(in_width * in_height * in_filters * sizeof(float));
    if(!result) return NULL;
    for (int i = 0; i < in_filters; i++)
    {
        for (int j = 0; j < in_width; j++)
        {
            for (int k = 0; k < in_height; k++)
            {
                result[in_width * in_height * i + in_height * j + k] = input[i][j][k];
            }
        }
    }
    return result;
}

float ***ADD_SKIP_CONNECTION(float ***input, float ***output, int width, int height, int channels)
{
    for (int i = 0; i < channels; i++)
    {
        for (int j = 0; j < width; j++)
        {
            for (int k = 0; k < height; k++)
            {
                output[i][j][k] += input[i][j][k];
            }
        }
    }
    return output;
}

void batchnorm_4d(float ***input, int height, int width, int num_channels, float *gamma, float *beta, float epsilon) {
    // For each channel
    for (int c = 0; c < num_channels; c++) {
        // Calculate mean for current channel
        float mean = 0.0;
        int count = 0;
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                mean += input[c][h][w];
                count++;
            }
        }
        mean /= (float)count;

        // Calculate variance for current channel
        float variance = 0.0;
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                float diff = input[c][h][w] - mean;
                variance += diff * diff;
            }
        }
        variance /= (float)count;

        // Calculate value for optimization
        float std = sqrt(variance + epsilon);
        float scale = gamma[c] / std;
        float shift = -mean * scale + beta[c];

        // batchnorm
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                input[c][h][w] = input[c][h][w] * scale + shift;
            }
        }
    }
}

void BatchNorm_free(float *gamma, float *beta)
{
    if (gamma)
        free(gamma);
    if (beta)
        free(beta);
}