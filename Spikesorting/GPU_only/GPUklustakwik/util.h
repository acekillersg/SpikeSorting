#pragma once
#ifndef UTIL__H__
#define UTIL__H__

#include<stdio.h>
//#include "log.h"

#include "cuda_runtime.h"  
#include "device_launch_parameters.h"

extern const float HugeScore;

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))

int irand(int min, int max);
FILE *fopen_safe(char *fname, char *mode);
void MatPrint(FILE *fp, float *Mat, int nRows, int nCols);
void CompareVectors(float *A, float *B, int N);
void gpuErrchk(cudaError_t);


//template <unsigned int blockSize>
//float reduceFlt(float *d_idata, unsigned int n);
//
//
//template <unsigned int blockSize>
//int reduceInt(int *d_idata, unsigned int n);
#endif