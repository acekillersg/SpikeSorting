#ifndef DLINALG__CUH__
#define DLINALG__CUH__

#include<vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace std;

// Matrix which has the form of a block matrix and a nonzero diagonal
// Unmasked gives the indices corresponding to the block matrix
// Masked gives the indices corresponding to the diagonal

class d_BlockPlusDiagonalMatrix
{
public:
    thrust::device_vector<float> Block;
    thrust::device_vector<float> Diagonal;
    thrust::device_vector<int> *Unmasked;
    thrust::device_vector<int> *Masked;
    int NumUnmasked, NumMasked;
    d_BlockPlusDiagonalMatrix(thrust::device_vector<int> &_Masked, thrust::device_vector<int> &_Unmasked);
//void compare(float *Flat);
};

int CusolverCholesky(d_BlockPlusDiagonalMatrix &In, d_BlockPlusDiagonalMatrix &Out);
void CublasTriSolve(d_BlockPlusDiagonalMatrix &M, thrust::device_vector<float> &x, thrust::device_vector<float> &Out);

#endif