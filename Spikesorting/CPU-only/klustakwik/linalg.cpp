#include "linalg.h"
#include<math.h>
#include<iostream>

using namespace std;

BlockPlusDiagonalMatrix::BlockPlusDiagonalMatrix(vector<int> &_Masked, vector<int> &_Unmasked)
{
	Masked = &_Masked;
	Unmasked = &_Unmasked;
	NumUnmasked = Unmasked->size();
	NumMasked = Masked->size();
	Block.resize(NumUnmasked*NumUnmasked);
	Diagonal.resize(NumMasked);
}

void BlockPlusDiagonalMatrix::compare(float *Flat)
{
	int nDims = NumUnmasked + NumMasked;
	float meanerr = 0.0;
	float maxerr = 0.0;
	int nerr = 0;
	int ntotal = 0;
	for (int ii = 0; ii < NumUnmasked; ii++)
	{
		int i = (*Unmasked)[ii];
		for (int jj = 0; jj < NumUnmasked; jj++)
		{
			int j = (*Unmasked)[jj];
			float x = Block[ii*NumUnmasked + jj];
			float y = Flat[i*nDims + j];
			float err = fabs(x - y);
			ntotal++;
			if (err > 0)
			{
				nerr++;
				meanerr += err;
				if (err > maxerr)
					maxerr = err;
			}
		}
	}
	for (int ii = 0; ii < NumMasked; ii++)
	{
		int i = (*Masked)[ii];
		float x = Diagonal[ii];
		float y = Flat[i*nDims + i];
		float err = fabs(x - y);
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
		cout << "Comparison error n=" << nerr << " (" << (100.0*nerr) / ntotal << "%), mean=" << meanerr << ", max=" << maxerr << endl;
	}
	else
		cout << "No comparison error." << endl;
}

// Cholesky Decomposition
// In provides upper triangle of input matrix (In[i*D + j] >0 if j>=i);
// which is the top half of a symmetric matrix
// Out provides lower triange of output matrix (Out[i*D + j] >0 if j<=i);
// such that Out' * Out = In.
// D is number of dimensions
//
// returns 0 if OK, returns 1 if matrix is not positive definite

int BPDCholesky(BlockPlusDiagonalMatrix &In, BlockPlusDiagonalMatrix &Out)
{
	int ii, jj, kk;
	float sum;
	int NumUnmasked = (int)In.NumUnmasked;

	// main bit for unmasked features
	for (ii = 0; ii < NumUnmasked; ii++)
	{
		for (jj = ii; jj < NumUnmasked; jj++)
		{
			sum = In.Block[ii*NumUnmasked + jj];

			for (kk = ii - 1; kk >= 0; kk--)
			{
				sum -= Out.Block[ii*NumUnmasked + kk] * Out.Block[jj*NumUnmasked + kk];
			}
			if (ii == jj) {
				if (sum <= 0) return(1); // Cholesky decomposition has failed
				Out.Block[ii*NumUnmasked + ii] = (float)sqrt(sum);
			}
			else {
				Out.Block[jj*NumUnmasked + ii] = sum / Out.Block[ii*NumUnmasked + ii];
			}
		}
	}
	// main bit for masked features
	for (ii = 0; ii < (int)In.NumMasked; ii++)
	{
		float sum = In.Diagonal[ii];
		if (sum <= 0)
			return 1; // Cholesky failed
		Out.Diagonal[ii] = (float)sqrt(sum);
	}

	return 0; // for success
}


// Solve a set of linear equations M*Out = x.
// Where M is lower triangular (M[i*D + j] >0 if j>=i);
// D is number of dimensions
void BPDTriSolve(BlockPlusDiagonalMatrix &M, vector<float> &x,
	vector<float> &Out)
{
	const int NumUnmasked = M.NumUnmasked;
	const int NumMasked = M.NumMasked;
	const float * __restrict ptr_x = &(x[0]);
	float * __restrict ptr_Out = &(Out[0]);
	if (NumUnmasked)
	{
		const int * __restrict Unmasked = &((*M.Unmasked)[0]);
		for (int ii = 0; ii < NumUnmasked; ii++)
		{
			const int i = Unmasked[ii];
			float sum = ptr_x[i];
			const float * __restrict row = &(M.Block[ii*M.NumUnmasked]);
			for (int jj = 0; jj < ii; jj++) // j<i
			{
				const int j = Unmasked[jj];
				sum += row[jj] * ptr_Out[j];
			}
			ptr_Out[i] = -sum / row[ii];
		}
	}
	if (NumMasked)
	{
		const int * __restrict Masked = &((*M.Masked)[0]);
		const float * __restrict Diagonal = &(M.Diagonal[0]);
		for (int ii = 0; ii < NumMasked; ii++)
		{
			const int i = Masked[ii];
			Out[i] = -ptr_x[i] / Diagonal[ii];
		}
	}
}