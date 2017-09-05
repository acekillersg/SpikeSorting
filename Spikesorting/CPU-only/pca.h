#pragma once
#pragma execution_character_set("utf-8")

#ifndef PCA__H__
#define PCA__H__

#include<iostream>
#include<algorithm>
#include<cstdlib>
#include<fstream>
#include "Eigen/Dense"
using namespace std;
using namespace Eigen;

class Pca{
private:
	static const int samples = 1000;
	static const int channels = 32;
	//MatrixXd X(10000, 128), C(128, 128);
public:
	void featurenormalize(MatrixXd &X);
	void computeCov(MatrixXd &X, MatrixXd &C);
	void computeEig(MatrixXd &C, MatrixXd &vec, MatrixXd &val);
	int computeDim(MatrixXd &val);
	void result(MatrixXd &C, int dim);
};
#endif