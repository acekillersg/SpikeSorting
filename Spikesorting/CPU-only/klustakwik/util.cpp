// Disable some Visual Studio warnings
#define _CRT_SECURE_NO_WARNINGS

#include "util.h"
#include<stdlib.h>
#include<stdarg.h>
#include<math.h>
#include<iostream>

const float HugeScore = (float)1e32;

/* integer random number between min and max*/
int irand(int min, int max)
{
	return (rand() % (max - min + 1) + min);
}

FILE *fopen_safe(char *fname, char *mode) {
	FILE *fp;
	fp = fopen(fname, mode);
	if (!fp) {
		fprintf(stderr, "Could not open file %s\n", fname);
		abort();
	}
	return fp;
}

// Print a matrix
void MatPrint(FILE *fp, float *Mat, int nRows, int nCols) {
	int i, j;

	for (i = 0; i<nRows; i++) {
		for (j = 0; j<nCols; j++) {
			fprintf(fp, "%.5g ", Mat[i*nCols + j]);
			//Output("%.5g ", Mat[i*nCols + j]);
		}
		fprintf(fp, "\n");
		//Output("\n");
	}
}

void CompareVectors(float *A, float *B, int N)
{
	float meanerr = 0.0;
	float maxerr = 0.0;
	int nerr = 0;
	int ntotal = 0;
	for (int i = 0; i < N; i++)
	{
		float err = fabs(A[i] - B[i]);
		ntotal++;
		if (err > 0)
		{
			nerr++;
			meanerr += err;
			if (err > maxerr)
				maxerr = err;
		}
	}
	if (nerr)
	{
		meanerr /= nerr;
		std::cout << "Comparison error n=" << nerr << " (" << (100.0*nerr) / ntotal << "%), mean=" << meanerr << ", max=" << maxerr << std::endl;
	}
	else
		std::cout << "No comparison error." << std::endl;
}