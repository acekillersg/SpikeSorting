#ifndef SPIKEDETECT__H__
#define SPIKEDETECT__H__

#include<iostream>
#include <map>
#include <vector>
#include <utility>
#include "parameters.h"
#include "Eigen/Dense"

//三次样条插值头文件
#include <iomanip>
#include <spline3interp.h>

using namespace std;
using namespace splab;

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
	//构造函数与析构函数
	SpikeDetect();
	SpikeDetect(std::string filename);
	~SpikeDetect();
	//测试输出
	void output();
	//对波形进行带通滤波
	void filter(int rate, float high, float low, int order);
	//计算阈值
	void threshold();
	//对波形进行转换
	void transform();
	void flood_fill(int x, int y, int label);
	//提取联通的点，组成波形
	void detect(std::map <int, std::vector < std::pair<int, int> >  > &comps);
	//float floodfilldetect();
	////根据高低阈值归一化电信号的电位值
	float normalize(float x);
	////根据componment在traces_t中提取wave,
	void comp_wave(int s_min, int s_max, std::vector < std::pair<int, int> > tmp_comp, std::vector<std::vector <float> > &wave);
	////根据wave和componment，计算mask
	void mask(int s_min, int s_max, std::vector<float > &mask, std::vector<int > &mask_bin, std::vector<std::vector <float> > &wave);
	////计算每个wave的中心时间,其中检测中心时间与GT的jetter<=12,就认为波形对照成功
	void spike_sample_aligned(int s_min, int s_max, float &s_aligned, std::vector<std::vector <float> > &wave);
	////根据wave中心对齐时间，提取waveform(要在此处去除死亡信道)
	void extract(int before, int after, float &s_aligned, std::vector <std::vector <float> > &waveform);
	////三次样条插值使得提取的波形平滑化
	void align(float s_aligned, int before, int after, std::vector <std::vector <float> > &waveform, std::vector <std::vector <float> > &wavealign);
	////计算waveforms的主成分
	void PCA(int n_spikes, std::vector< std::vector< std::vector <float> > >&waveforms, std::vector <std::vector <float> > &masks, std::vector< std::vector< std::vector <float> > >&out);
};
#endif
