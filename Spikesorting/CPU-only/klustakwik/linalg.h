#ifndef LINALG__H__
#define LINALG__H__

#include<vector>

using namespace std;

// Matrix which has the form of a block matrix and a nonzero diagonal
// Unmasked gives the indices corresponding to the block matrix
// Masked gives the indices corresponding to the diagonal
class BlockPlusDiagonalMatrix
{
public:
	vector<float> Block;
	vector<float> Diagonal;
	vector<int> *Unmasked;
	vector<int> *Masked;
	int NumUnmasked, NumMasked;
	BlockPlusDiagonalMatrix(vector<int> &_Masked, vector<int> &_Unmasked);
	void compare(float *Flat);
};

int BPDCholesky(BlockPlusDiagonalMatrix &In, BlockPlusDiagonalMatrix &Out);
void BPDTriSolve(BlockPlusDiagonalMatrix &M, vector<float> &x, vector<float> &Out);

#endif
