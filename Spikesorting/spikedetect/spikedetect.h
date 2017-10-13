#ifndef SPIKEDETECT__H__
#define SPIKEDETECT__H__

#include<iostream>
#include <map>
#include <vector>
#include <utility>
#include "parameters.h"
#include "Eigen/Dense"

//����������ֵͷ�ļ�
#include <iomanip>
#include <spline3interp.h>

using namespace std;
using namespace splab;

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
	//���캯������������
	SpikeDetect();
	SpikeDetect(std::string filename);
	~SpikeDetect();
	//�������
	void output();
	//�Բ��ν��д�ͨ�˲�
	void filter(int rate, float high, float low, int order);
	//������ֵ
	void threshold();
	//�Բ��ν���ת��
	void transform();
	void flood_fill(int x, int y, int label);
	//��ȡ��ͨ�ĵ㣬��ɲ���
	void detect(std::map <int, std::vector < std::pair<int, int> >  > &comps);
	//float floodfilldetect();
	////���ݸߵ���ֵ��һ�����źŵĵ�λֵ
	float normalize(float x);
	////����componment��traces_t����ȡwave,
	void comp_wave(int s_min, int s_max, std::vector < std::pair<int, int> > tmp_comp, std::vector<std::vector <float> > &wave);
	////����wave��componment������mask
	void mask(int s_min, int s_max, std::vector<float > &mask, std::vector<int > &mask_bin, std::vector<std::vector <float> > &wave);
	////����ÿ��wave������ʱ��,���м������ʱ����GT��jetter<=12,����Ϊ���ζ��ճɹ�
	void spike_sample_aligned(int s_min, int s_max, float &s_aligned, std::vector<std::vector <float> > &wave);
	////����wave���Ķ���ʱ�䣬��ȡwaveform(Ҫ�ڴ˴�ȥ�������ŵ�)
	void extract(int before, int after, float &s_aligned, std::vector <std::vector <float> > &waveform);
	////����������ֵʹ����ȡ�Ĳ���ƽ����
	void align(float s_aligned, int before, int after, std::vector <std::vector <float> > &waveform, std::vector <std::vector <float> > &wavealign);
	////����waveforms�����ɷ�
	void PCA(int n_spikes, std::vector< std::vector< std::vector <float> > >&waveforms, std::vector <std::vector <float> > &masks, std::vector< std::vector< std::vector <float> > >&out);
};
#endif
