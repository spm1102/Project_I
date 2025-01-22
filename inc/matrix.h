#ifndef MATRIX
#define MATRIX

#include<stdio.h>
#include<stdlib.h>


float **MATRIX_Create(int rows, int cols);
float **MATRIX_MatMul(float **p_matrix1, int p_matrix1_rows, int p_matrix1_cols, float **p_matrix2, int p_matrix2_rows, int p_matrix2_cols);
float MATRIX_ElemMul(float **p_matrix1, float **p_matrix2, int rows, int cols);
void MATRIX_Print(float **matrix, int rows, int cols);
void MATRIX_Free(float **kernel, int width, int height);



#endif