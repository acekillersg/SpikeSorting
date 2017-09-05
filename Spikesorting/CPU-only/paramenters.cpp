//#pragma once
//#pragma execution_character_set("utf-8")

#include "paramenters.h"

Params :: Params(){
	// filter
	filter_low = 500;
	filter_high_factor = 0.95 * .5;
	filter_butter_order = 3;

	// Data chunks
	chunk_size_seconds = 1.;
	chunk_overlap_seconds = 0.015;


	// Threshold
	n_excerpts = 50;
	excerpt_size_seconds = 1.;
	use_single_threshold = 'True';
	threshold_strong_std_factor = 4.5;
	threshold_weak_std_factor = 2.;
	//    1:positive,    0:both   -1:negative
	detect_spikes_symbol = -1;

	// Connected components
	connected_component_join_size = 1;

	//Spike extractions
	extract_s_before = 10;
	extract_s_after = 10;
	weight_power = 2;

	// Features
	n_features_per_channel = 3;
	pca_n_waveforms_max = 10000;
}

Params::~Params(){

}

/**************************************Get the paramenter's value*******************************/
//Filter
float Params::get_filter_low(){ return filter_low; }
float Params::get_filter_high_factor(){ return filter_high_factor; }
int Params::get_filter_butter_order(){ return filter_butter_order; }

// Data chunks
float Params::get_chunk_size_seconds(){ return chunk_size_seconds; }
float Params::get_chunk_overlap_seconds(){ return chunk_overlap_seconds; }

// Threshold
int Params::get_n_excerpts(){ return n_excerpts; }
float Params::get_excerpt_size_seconds(){ return excerpt_size_seconds; }
bool Params::get_use_single_threshold(){ return use_single_threshold; }
float Params::get_threshold_strong_std_factor(){ return threshold_strong_std_factor; }
float Params::get_threshold_weak_std_factor(){ return threshold_weak_std_factor; }
//   detect_spikes_symbol:(1:positive,    0:both   -1:negative)
int Params::get_detect_spikes_symbol(){ return detect_spikes_symbol; }

// Connected components
int Params::get_connected_component_join_size(){ return connected_component_join_size; }

//Spike extractions
int Params::get_extract_s_before(){ return extract_s_before; }
int Params::get_extract_s_after(){ return extract_s_after; }
int Params::get_weight_power(){ return weight_power; }

// Features
int Params::get_n_features_per_channel(){ return n_features_per_channel; }
int Params::get_pca_n_waveforms_max(){ return pca_n_waveforms_max; }

/**************************************Set the paramenter's value*******************************/
void Params::set_filter_low(float n){ filter_low = n; }
void Params::set_filter_high_factor(float n){ filter_high_factor = n; }
void Params::set_filter_butter_order(int m){ filter_butter_order = m; }

// Data chunks
void Params::set_chunk_size_seconds(float n){ chunk_size_seconds= n; }
void Params::set_chunk_overlap_seconds(float n){ chunk_overlap_seconds= n; }

// Threshold
void Params::set_n_excerpts(int m){ n_excerpts = m; }
void Params::set_excerpt_size_seconds(float n){ excerpt_size_seconds = n; }
void Params::set_use_single_threshold(bool x){ use_single_threshold = x; }
void Params::set_threshold_strong_std_factor(float n){threshold_strong_std_factor = n; }
void Params::set_threshold_weak_std_factor(float n){threshold_weak_std_factor = n; }
//   detect_spikes_symbol:(1:positive,    0:both   -1:negative)
void Params::set_detect_spikes_symbol(int m){detect_spikes_symbol = m; }

// Connected components
void Params::set_connected_component_join_size(int m){ connected_component_join_size = m; }

//Spike extractions
void Params::set_extract_s_before(int m){ extract_s_before = m; }
void Params::set_extract_s_after(int m){ extract_s_after = m; }
void Params::set_weight_power(int m){ weight_power = m; }

// Features
void Params::set_n_features_per_channel(int m){ n_features_per_channel = m; }
void Params::set_pca_n_waveforms_max(int m){ pca_n_waveforms_max= m; }