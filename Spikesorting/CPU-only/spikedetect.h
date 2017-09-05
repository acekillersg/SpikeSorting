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
	//traces:在最初的函数中表示原始波形，经过Filter变为滤波后的波形，经过transform变为traces_t波形，
	//traces_t用于后面waveform中波形提取等一系列操作
	float traces[samples][channels];
	float traces_t[samples][channels];
	//crossing数组中，如果某点高于高阈值，则值为2，若高于低阈值，则值为1，低于低阈值，值为0
	int crossing[samples][channels];
	//spikedetect的参数文件
    Params p;
	//高低阈值
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
	//三次样条插值
	void align();
	void PCA(int n_spikes, std::vector< std::vector< std::vector <float> > >&waveforms, std::vector <std::vector <float> > &masks, std::vector< std::vector< std::vector <float> > >&out);
};
#endif