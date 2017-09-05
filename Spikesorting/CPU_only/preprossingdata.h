#pragma once
#pragma execution_character_set("utf-8")

#ifndef PREPROSSINGDATA__H__
#define PREPROSSINGDATA__H__

#include<iostream>
#include<vector>

class Preprossingdata{
private:
	int n_spikes;
	int  num_features;
	std::vector <std::vector<float> > features;
	std::vector <std::vector<float> > masks;
public:
	Preprossingdata(int n_spikes_, int n_channels_, int n_pca, std::vector< std::vector< std::vector <float> > >&waveforms, std::vector <std::vector <float> > &masks);
	~Preprossingdata();

	void rawsparsedata(std::vector<float > &all_features, std::vector<float > &all_masks, std::vector<int > &all_unmasked, std::vector<int > &offsets,
		std::vector<float > &noise_mean, std::vector<float > &noise_variance, std::vector<float > &correction_terms, std::vector<float > &float_num_unmasked);
	void precomputations();
};
#endif