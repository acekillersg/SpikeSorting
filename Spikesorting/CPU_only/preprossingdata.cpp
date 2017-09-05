#pragma once
#pragma execution_character_set("utf-8")

#include"preprossingdata.h"

Preprossingdata::Preprossingdata(int n_spikes_, int n_channels_, int n_pca, std::vector< std::vector< std::vector <float> > >&out, std::vector <std::vector <float> > &masks_){
	n_spikes = n_spikes_;
	num_features = n_channels_ * n_pca;
	for (int i = 0; i < n_spikes; i++)
	{
		std::vector<float > tmp;
		for (int j = 0; j < n_channels_; j++)
			for (int k = 0; k < n_pca; k++)
				tmp.push_back(out[i][j][k]);
		features.push_back(tmp);

		for (int j = 0; j < n_channels_; j++)
			for (int k = 0; k < n_pca; k++)
				tmp.push_back(masks_[i][j]);
		masks.push_back(tmp);
	}
}

Preprossingdata::~Preprossingdata(){}

void Preprossingdata::rawsparsedata(std::vector<float > &all_features, std::vector<float > &all_masks, std::vector<int > &all_unmasked,
	std::vector<int > &offsets, std::vector<float > &noise_mean, std::vector<float > &noise_variance, std::vector<float > &correction_terms,
	std::vector<float > &float_num_unmasked){
	//����ȡ��features�����У����������й�һ��
	for (int j = 0; j < num_features; j++)
	{
		float min = features[0][j], max = features[0][j];
		for (int i = 1; i < n_spikes; i++){
			if (min > features[i][j]) min = features[i][j];
			if (max < features[i][j]) max = features[i][j];
		}
		for (int i = 1; i < n_spikes; i++){
			float t = max > min ? (max - min) : 1.0;
			features[i][j] = (features[i][j] - min) / t;
		}
	}
	//����masks��������δ���ڱε�������ȡ�������ڱε���������fetsum�Ȳ��������ڼ���������ֵ�ͷ������all_masks��all_unmasked��offsets
	std::vector<float > fetsum(num_features, 0.0);
	std::vector<float > fet2sum(num_features, 0.0);
	std::vector<int > nsum(num_features, 0);
	int curoff = 0;
	offsets.push_back(curoff);
	for (int i = 0; i < n_spikes; i++){
		for (int j = 0; j < num_features; j++){
			if (masks[i][j]> 0)
			{
				curoff++;
				all_features.push_back(features[i][j]);
				all_masks.push_back(masks[i][j]);
				all_unmasked.push_back(j);
			}
			else {
				fetsum[j] += features[i][j];
				fet2sum[j] += features[i][j] * features[i][j];
				nsum[j] += 1;
			}
			offsets.push_back(curoff);
		}
	}
	//���������ľ�ֵ�ͷ������noise_mean��noise_variance
	for (int i = 0; i < num_features; i++)
	{
		if (nsum[i] == 0) nsum[i] = 1;
		noise_mean[i] = fetsum[i] / nsum[i];
		noise_variance[i] = fet2sum[i] / nsum[i] - noise_mean[i] * noise_mean[i];
	}
	//ͨ��������ֵ��������ڱ�����������features��ÿ��δ���ڱεĵ������,����ȣ�����all_features��correction_terms
	std::vector<float > fet_temp(curoff, 0.0);
	for (int i = 0; i < curoff; i++)
	{
		fet_temp[i] = all_features[i] * all_masks[i] + (1 - all_masks[i]) * noise_mean[i];
		correction_terms[i] = all_masks[i] * all_features[i] * all_features[i] + (1 - all_masks[i]) *(noise_mean[i] * noise_mean[i] + noise_variance[i]) - fet_temp[i] * fet_temp[i];
		all_features[i] = fet_temp[i];
	}
	//��all_unmasked�����Ż������offsets����δ���ڱε���ͬ��spike���������ϲ�������unmasked��u_start��u_end

	//ͨ��all_masks��offsets����ÿ��spike������Ϊ���ڱε�masks������ӣ�����float_num_unmasked
	for (int i = 0; i < offsets.size() - 1; i++)
		for (int j = offsets[i]; j < offsets[i + 1]; j++)
			float_num_unmasked[i] += all_masks[j];
}