#pragma once
#pragma execution_character_set("utf-8")

#include <iostream>
#include<fstream>
#include "parameters.h"
#include "spikedetect.h"
#include "probe.h"
#define CHANNELS 32
#define SAMPLES 1000

using namespace std;
using namespace splab;

int main()
{
	Params p;
	/*------------------------------spikedetect.h-------------------------------------------*/
	SpikeDetect s("traces_f.txt");
	s.threshold();
	s.transform();
	std::map <int, std::vector < std::pair<int, int> >  > comps;
	s.detect(comps);
	std::vector < std::pair<int, int> > tmp;
	std::vector< std::vector< std::vector <float> > > waveforms;
	std::vector< std::vector <float> > masks;
	std::vector <float> aligns;
	int it = 0;
	for (const auto &w : comps){
		it++;
		tmp = w.second;
		////表示采样时间的范围[s_min,s_max]
		int s_min = tmp[0].first;
		int s_max = tmp[0].first;
	
		////表示component被掩蔽的信道,掩蔽为0，未掩蔽为1
		std::vector<int > mask_bin(CHANNELS, 0);
		mask_bin[tmp[0].second] = 1;
		for (unsigned int ii = 1; ii < tmp.size(); ii++){
			if (s_min > tmp[ii].first) s_min = tmp[ii].first;
			if (s_max < tmp[ii].first) s_max = tmp[ii].first;
			mask_bin[tmp[ii].second] = 1;
		}

		////提取waveform
		int w = s_max - s_min + 1;
		vector <vector <float> > wave(w);
		for (int i = 0; i < w; i++)
			wave[i].resize(CHANNELS, 0);
		s.comp_wave(s_min, s_max, tmp,wave);

		////计算掩蔽向量
	    std::vector<float > mask(CHANNELS, 0.0);//未被掩蔽信道的峰值归一化后的值
        s.mask(s_min, s_max,mask, mask_bin,wave);
		masks.push_back(mask);

		////计算中心对齐时间
		float s_aligned;
		s.spike_sample_aligned(s_min, s_max, s_aligned,wave);
		aligns.push_back(s_aligned);

		////根据中心时间提取标准化波形
		int width = p.get_extract_s_before() + p.get_extract_s_after() + 3;
		std::vector <std::vector <float> > waveform(width);
		for (int i = 0; i < width; i++)
			waveform[i].resize(CHANNELS,0);
		s.extract(p.get_extract_s_before(), p.get_extract_s_after(), s_aligned, waveform);

		////对waveform进行三次样条插值，并将35个点抽样成32个点
		int len = p.get_extract_s_before() + p.get_extract_s_after();
		std::vector <std::vector <float> > wavealign(len);
		for (int i = 0; i < len; i++)
			wavealign[i].resize(CHANNELS, 0);
		s.align(s_aligned, p.get_extract_s_before(), p.get_extract_s_after(), waveform, wavealign);
		waveforms.push_back(wavealign);
	}

	/*ofstream fout("waveforms.txt");
	for (int i = 0; i < 7; i++)
	{
		for (int j = 0; j <32; j++)
		{
			fout << waveforms[i][j][0];
			for (int k = 1; k < 32; k++)
				fout <<" "<<  waveforms[i][j][k];
			fout << endl;
		}
		//fout << endl;
	}
	fout.close();*/
	cout << "============================================" << endl;
	/*for (int i = 0; i < 2; i++){
		for (int j = 0; j < 35; j++){
			for (int k = 0; k < 32; k++)
				cout << waveforms[i][j][k] << "  ";
			cout << endl;
		}
		cout << endl << endl;;
	}*/
		
	////对插值之后的waveforms进行提取特征
	int n_spikes = comps.size();
	int n_pca = p.get_n_features_per_channel();
	std::vector< std::vector< std::vector <float> > > out(n_spikes);
	for (int i = 0; i < n_spikes; i++){
		out[i].resize(CHANNELS);
		for (int j = 0; j < CHANNELS; j++)
			out[i][j].resize(n_pca);
	}
	s.PCA(n_spikes, waveforms, masks, out);

	system("Pause");
}