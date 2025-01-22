#include "matrix.h"

float **MATRIX_Create(int rows, int cols)
{
    float **p_matrix = (float **)malloc(rows * sizeof(float *));
    if (!p_matrix)
        return NULL;
    for (int i = 0; i < rows; i++)
    {
        p_matrix[i] = (float *)malloc(cols * sizeof(float));
        if (!p_matrix[i])
            return NULL;
    }
    return p_matrix;
}

float **MATRIX_MatMul(float **p_matrix1, int p_matrix1_rows, int p_matrix1_cols, float **p_matrix2, int p_matrix2_rows, int p_matrix2_cols)
{
    float **result = MATRIX_Create(p_matrix1_rows, p_matrix2_cols);

    for (int i = 0; i < p_matrix1_rows; i++)
    {
        for (int j = 0; j < p_matrix2_cols; j++)
        {
            result[i][j] = 0;
            for (int k = 0; k < p_matrix2_rows; k++)
            {
                result[i][j] += p_matrix1[i][k] * p_matrix2[k][j];
            }
        }
    }

    return result;
}

float MATRIX_ElemMul(float **p_matrix1, float **p_matrix2, int rows, int cols)
{
    float result = 0;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            result += p_matrix1[i][j] * p_matrix2[i][j];
        }
    }
    return result;
}

void MATRIX_Free(float **matrix, int width, int height)
{
    for (int i = 0; i < width; i++)
    {
        free(matrix[i]);
    }
    free(matrix);
}

void MATRIX_Print(float **matrix, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%.3lf ", matrix[i][j]);
        }
        printf("\n");
    }
}