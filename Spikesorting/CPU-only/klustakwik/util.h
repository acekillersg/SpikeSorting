#ifndef UTIL__H__
#define UTIL__H__

#include<stdio.h>
//#include "log.h"

extern const float HugeScore;

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))

int irand(int min, int max);
FILE *fopen_safe(char *fname, char *mode);
void MatPrint(FILE *fp, float *Mat, int nRows, int nCols);
void CompareVectors(float *A, float *B, int N);

#endif