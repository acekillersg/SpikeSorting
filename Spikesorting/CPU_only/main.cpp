#pragma once
#pragma execution_character_set("utf-8")

#include <iostream>
#include "paramenters.h"
#include "spikedetect.h"
#include "probe.h"
#include "preprossingdata.h"
#include "klustakwik.h"
#define CHANNELS 32
#define SAMPLES 1000
using namespace std;

int main()
{
	/*-----------------------------test_paramenter.h-------------------------------------------*/
	Params p;
	std::cout << p.get_chunk_overlap_seconds() << " " << p.get_chunk_size_seconds() << std::endl;
	/*--------------------------------------------------------------------------------------------*/
	/*------------------------------------test_probe.h-------------------------------------------*/
	Probe prb;
	map <int, vector < int >  > probe_adjacency_list;
	prb.edges_to_adjacency_list(probe_adjacency_list);
	prb.test_output();
	//去除死亡信道
	vector<int> dead_ch(CHANNELS, 0);
	prb.dead_channels(dead_ch, CHANNELS);
	for (int i = 0; i < CHANNELS; i++)
		cout << dead_ch[i] << " ";
	cout << endl;
	/*--------------------------------------------------------------------------------------------*/
	/*------------------------------test_spikedetect.h-------------------------------------------*/
	SpikeDetect s("traces_f.txt");
	s.threshold();
	s.transform();
	std::map <int, std::vector < std::pair<int, int> >  > comps;
	s.detect(comps);
	std::cout << "=============line1=================" << std::endl;
	std::vector < std::pair<int, int> > tmp;

	std::vector< std::vector< std::vector <float> > > waveforms;
	std::vector< std::vector <float> > masks;
	for (unsigned int i = 0; i < comps.size(); i++)
	{
		tmp = comps[i];
		//表示采样时间的范围[s_min,s_max]
		int s_min = tmp[0].first;
		int s_max = tmp[0].first;
		//表示component被掩蔽的信道,掩蔽为0，未掩蔽为1
		std::vector<int > mask_bin(CHANNELS, 0);
		mask_bin[tmp[0].second] = 1;
		for (unsigned int ii = 1; ii < tmp.size(); ii++){
			if (s_min > tmp[ii].first) s_min = tmp[ii].first;
			if (s_max < tmp[ii].first) s_max = tmp[ii].first;
			mask_bin[tmp[ii].second] = 1;
		}
		//提取waveform
		std::vector <std::vector <float> > wave((s_max - s_min), std::vector<float>(CHANNELS, 0));
		s.comp_wave(s_min, s_max, tmp, wave);
		//计算掩蔽向量
		std::vector<float > mask(CHANNELS, 0.0);//未被掩蔽信道的峰值归一化后的值
		s.mask(mask, mask_bin, wave);
		masks.push_back(mask);
		//计算中心对齐时间
		float s_aligned;
		s.spike_sample_aligned(s_min, s_max, wave, s_aligned);
		//根据中心时间提取标准化波形
		std::vector <std::vector <float> > waveform(p.get_extract_s_before() + p.get_extract_s_after(), vector<float>(CHANNELS, 0));
		s.extract(s_aligned, waveform);
		waveforms.push_back(waveform);
	}
	//PCA 提取波形的主成分
	int n_spikes = comps.size();
	int n_pca = p.get_n_features_per_channel();
	std::vector< std::vector< std::vector <float> > > out(n_spikes, std::vector<std::vector<float> >(CHANNELS, vector<float>(n_pca)));
	s.PCA(n_spikes, waveforms, masks, out);
	//s.output();

	/*-------------------------------------preprossingdata.h部分------------------------------------*/
	Preprossingdata data(n_spikes, CHANNELS, n_pca, out, masks);

	std::vector<float > all_features;
	std::vector<float > all_masks;
	std::vector<int > all_unmasked;
	std::vector<int >  offsets;
	std::vector<float >noise_mean;
	std::vector<float > noise_variance;
	std::vector<float > correction_terms;
	std::vector<float > float_num_unmasked(n_spikes, 0.0);
	data.rawsparsedata(all_features, all_masks, all_unmasked,
		offsets, noise_mean, noise_variance, correction_terms, float_num_unmasked);

	system("Pause");
}

//#include<iostream>
//#include<algorithm>
//#include<cstdlib>
//#include<fstream>
//#include "Eigen/Dense"
//using namespace std;
//using namespace Eigen;
//void featurenormalize(MatrixXf &X)
//{
//	计算每一维度均值
//	MatrixXf meanval = X.colwise().mean();
//	RowVectorXf meanvecRow = meanval;
//	样本均值化为0
//	X.rowwise() -= meanvecRow;
//}
//void computeCov(MatrixXf &X, MatrixXf &covMat)
//{
//	计算协方差矩阵C = XTX /(n-1);
//	covMat = X.adjoint() * X;
//	covMat = covMat.array() / (X.rows() - 1);
//}
//void computeEig(MatrixXf &C, MatrixXf &vec, MatrixXf &val)
//{
//	计算特征值和特征向量，使用selfadjont按照对阵矩阵的算法去计算，可以让产生的vec和val按照有序排列
//	SelfAdjointEigenSolver<MatrixXf> eig(C);
//	vec = eig.eigenvectors();
//	val = eig.eigenvalues();
//}
//int computeDim(MatrixXf &val)
//{
//	int dim;
//	double sum = 0;
//	for (int i = val.rows() - 1; i >= 0; --i)
//	{
//		sum += val(i, 0);
//		dim = i;
//
//		if (sum / val.sum() >= 0.95)
//			break;
//	}
//	return val.rows() - dim;
//}
//int main()
//{
//	ifstream fin("input.txt");
//	ofstream fout("output.txt");
//	const int m = 2, n = 10;
//	MatrixXf X(2, 10), C(10, 10);
//	MatrixXf vec, val;
//
//	读取数据
//	double in[200];
//	for(int i = 0; i < m; ++i)
//	{
//		for (int j = 0; j < n; ++j){
//			fin >> in[j];
//			X(i, j) = in[j];
//		}
//	}
//	pca
//
//	零均值化
//	featurenormalize(X);
//	计算协方差
//	cout << X << endl;
//	computeCov(X, C);
//	cout << C << endl;
//	计算特征值和特征向量
//	computeEig(C, vec, val);
//	cout << vec << endl << val;
//	计算损失率，确定降低维数
//	int dim = computeDim(val);
//	计算结果
//	MatrixXf res = X * vec.rightCols(3);
//	cout << vec.rightCols(3) << endl;
//	输出结果
//	cout << "the result is " << res.rows() << "x" << res.cols() << " after pca algorithm." << endl;
//	cout << res;
//	system("pause");
//	return 0;
//}
