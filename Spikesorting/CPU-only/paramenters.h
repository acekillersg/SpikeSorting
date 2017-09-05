//#pragma once
//#pragma execution_character_set("utf-8")

#include <iostream>
#include <cstring>
#ifndef PARAMENTERS__H__
#define PARAMENTERS__H__
class Params{
private:
	// filter
	float filter_low ;
	float filter_high_factor ;
	int filter_butter_order ;

	// Data chunks
	float chunk_size_seconds ;
	float chunk_overlap_seconds;


	// Threshold
	int n_excerpts ;
	float excerpt_size_seconds;
	bool use_single_threshold ;
	float threshold_strong_std_factor;
	float threshold_weak_std_factor;
	//   detect_spikes_symbol:(1:positive,    0:both   -1:negative)
	int detect_spikes_symbol ;

	// Connected components
	int connected_component_join_size ;

	//Spike extractions
	int extract_s_before ;
	int extract_s_after ;
	int weight_power ;

	// Features
	int n_features_per_channel ;
	int pca_n_waveforms_max ;

	//Class define functions
public:
	Params();
	~Params();
	/**************************************Get the paramenter's value*******************************/
	float get_filter_low();
	float get_filter_high_factor();
	int get_filter_butter_order();

	// Data chunks
	float get_chunk_size_seconds();
	float get_chunk_overlap_seconds();

	// Threshold
	int get_n_excerpts();
	float get_excerpt_size_seconds();
	bool get_use_single_threshold();
	float get_threshold_strong_std_factor();
	float get_threshold_weak_std_factor();
	//   detect_spikes_symbol:(1:positive,    0:both   -1:negative)
	int get_detect_spikes_symbol();

	// Connected components
	int get_connected_component_join_size();

	//Spike extractions
	int get_extract_s_before();
	int get_extract_s_after();
	int get_weight_power();

	// Features
	int get_n_features_per_channel();
	int get_pca_n_waveforms_max();

	/**************************************Set the paramenter's value*******************************/
	void set_filter_low(float n);
	void set_filter_high_factor(float n);
	void set_filter_butter_order(int m);

	// Data chunks
	void set_chunk_size_seconds(float n);
	void set_chunk_overlap_seconds(float n);

	// Threshold
	void set_n_excerpts(int m);
	void set_excerpt_size_seconds(float n);
	void set_use_single_threshold(bool x);
	void set_threshold_strong_std_factor(float n);
	void set_threshold_weak_std_factor(float n);
	//   detect_spikes_symbol:(1:positive,    0:both   -1:negative)
	void set_detect_spikes_symbol(int m);

	// Connected components
	void set_connected_component_join_size(int m);

	//Spike extractions
	void set_extract_s_before(int m);
	void set_extract_s_after(int m);
	void set_weight_power(int m);

	// Features
	void set_n_features_per_channel(int m);
	void set_pca_n_waveforms_max(int m);
};
#endif
