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
	//ȥ�������ŵ�
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
		//��ʾ����ʱ��ķ�Χ[s_min,s_max]
		int s_min = tmp[0].first;
		int s_max = tmp[0].first;
		//��ʾcomponent���ڱε��ŵ�,�ڱ�Ϊ0��δ�ڱ�Ϊ1
		std::vector<int > mask_bin(CHANNELS, 0);
		mask_bin[tmp[0].second] = 1;
		for (unsigned int ii = 1; ii < tmp.size(); ii++){
			if (s_min > tmp[ii].first) s_min = tmp[ii].first;
			if (s_max < tmp[ii].first) s_max = tmp[ii].first;
			mask_bin[tmp[ii].second] = 1;
		}
		//��ȡwaveform
		std::vector <std::vector <float> > wave((s_max - s_min), std::vector<float>(CHANNELS, 0));
		s.comp_wave(s_min, s_max, tmp, wave);
		//�����ڱ�����
		std::vector<float > mask(CHANNELS, 0.0);//δ���ڱ��ŵ��ķ�ֵ��һ�����ֵ
		s.mask(mask, mask_bin, wave);
		masks.push_back(mask);
		//�������Ķ���ʱ��
		float s_aligned;
		s.spike_sample_aligned(s_min, s_max, wave, s_aligned);
		//��������ʱ����ȡ��׼������
		std::vector <std::vector <float> > waveform(p.get_extract_s_before() + p.get_extract_s_after(), vector<float>(CHANNELS, 0));
		s.extract(s_aligned, waveform);
		waveforms.push_back(waveform);
	}
	//PCA ��ȡ���ε����ɷ�
	int n_spikes = comps.size();
	int n_pca = p.get_n_features_per_channel();
	std::vector< std::vector< std::vector <float> > > out(n_spikes, std::vector<std::vector<float> >(CHANNELS, vector<float>(n_pca)));
	s.PCA(n_spikes, waveforms, masks, out);
	//s.output();

	/*-------------------------------------preprossingdata.h����------------------------------------*/
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
//	����ÿһά�Ⱦ�ֵ
//	MatrixXf meanval = X.colwise().mean();
//	RowVectorXf meanvecRow = meanval;
//	������ֵ��Ϊ0
//	X.rowwise() -= meanvecRow;
//}
//void computeCov(MatrixXf &X, MatrixXf &covMat)
//{
//	����Э�������C = XTX /(n-1);
//	covMat = X.adjoint() * X;
//	covMat = covMat.array() / (X.rows() - 1);
//}
//void computeEig(MatrixXf &C, MatrixXf &vec, MatrixXf &val)
//{
//	��������ֵ������������ʹ��selfadjont���ն��������㷨ȥ���㣬�����ò�����vec��val������������
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
//	��ȡ����
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
//	���ֵ��
//	featurenormalize(X);
//	����Э����
//	cout << X << endl;
//	computeCov(X, C);
//	cout << C << endl;
//	��������ֵ����������
//	computeEig(C, vec, val);
//	cout << vec << endl << val;
//	������ʧ�ʣ�ȷ������ά��
//	int dim = computeDim(val);
//	������
//	MatrixXf res = X * vec.rightCols(3);
//	cout << vec.rightCols(3) << endl;
//	������
//	cout << "the result is " << res.rows() << "x" << res.cols() << " after pca algorithm." << endl;
//	cout << res;
//	system("pause");
//	return 0;
//}
