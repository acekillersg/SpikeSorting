#pragma once
#pragma execution_character_set("utf-8")

#include<iostream>
#include <map>
#include <vector>
#include <utility>
#include "paramenters.h"
#include "Eigen/Dense"
#ifndef SPIKEDETECT__H__
#define SPIKEDETECT__H__

class SpikeDetect{
private:
	static const int samples = 1000;
	static const int channels = 32;
	//traces:������ĺ����б�ʾԭʼ���Σ�����Filter��Ϊ�˲���Ĳ��Σ�����transform��Ϊtraces_t���Σ�
	//traces_t���ں���waveform�в�����ȡ��һϵ�в���
	float traces[samples][channels];
	float traces_t[samples][channels];
	//crossing�����У����ĳ����ڸ���ֵ����ֵΪ2�������ڵ���ֵ����ֵΪ1�����ڵ���ֵ��ֵΪ0
	int crossing[samples][channels];
	//spikedetect�Ĳ����ļ�
    Params p;
	//�ߵ���ֵ
	float high_crossing, low_crossing;
public:
	SpikeDetect();
	SpikeDetect(std::string filename);
	~SpikeDetect();
	void output();
	void filter(int rate, float high, float low, int order);
	void threshold();
	void transform();
	void flood_fill(int x,int y,int label);
	//map <int, vector < pair<int, int> >  > & detect(std::map <int, std::vector < std::pair<int, int> >  > &comps);
	void detect(std::map <int, std::vector < std::pair<int, int> >  > &comps);
	float floodfilldetect();
	float normalize(float x);
	void comp_wave(int s_min, int s_max, std::vector < std::pair<int, int> > tmp_comp, std::vector<std::vector<float> > &wave);
	void mask(std::vector<float > &mask, std::vector<int > &mask_bin, std::vector<std::vector<float> > &wave);
	void spike_sample_aligned(int s_min, int s_max, std::vector<std::vector<float> > &wave, float &s_aligned);
	void extract(float &s_aligned, std::vector <std::vector <float> > &waveform);
	//����������ֵ
	void align();
	void PCA(int n_spikes, std::vector< std::vector< std::vector <float> > >&waveforms, std::vector <std::vector <float> > &masks, std::vector< std::vector< std::vector <float> > >&out);
};
#endif