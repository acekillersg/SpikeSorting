#pragma once
#pragma execution_character_set("utf-8")

#include <stdio.h>


struct spikedetekt_prm{
    //Filter
	float filter_low = 500.;
	float filter_high_factor = 0.95 * .5;//will be multiplied by the sample rate
	int filter_butter_order = 3;

	//Data chunks.
	float chunk_size_seconds = 1.;
	float chunk_overlap_seconds = .015;

	//Threshold.
	int n_excerpts = 50;
	float excerpt_size_seconds = 1.;
	bool use_single_threshold = true;
	float strong_factor = 4.5;
	float weak_factor = 2.;
	//char detect_spikes[10] = "negative";

	//Connected components.
	int connected_component_join_size = 1;

	//Spike extractions.
	int extract_s_before = 10;
	int extract_s_after = 10;
	int weight_power = 2;

	//Features.
	int n_features_per_channel = 3;
	int pca_n_waveforms_max = 10000;
};