#pragma once
#pragma execution_character_set("utf-8")

#ifndef KWIKPARAMETER__H__
#define KWIKPARAMETER__H__
#include <cmath>
class Params1{
public:
	int prior_point; //？？？
	int mua_point;
	int noise_point;
	int points_for_cluster_mask; //？？？
	float penalty_k; //用于计算penalty
	float penalty_k_log_n; //用于计算penalty
	int max_iterations;
	int num_starting_clusters;
	bool use_noise_cluster;
	bool use_mua_cluster;
	int num_changed_threshold; //？？？
	int full_step_every; //？？？
	int split_first;
	int split_every;
	int MaxPossibleClusters;
	float dist_thresh; //？？？
	int max_quick_step_candidates; //this uses around 760 MB RAM  ##？？？
	float max_quick_step_candidates_fraction; //？？？
	bool always_split_bimodal; //？？？
	float subset_break_fraction; //？？？
	float break_fraction; //？？？
	bool fast_split; //？？？
	//max_split_iterations = None, ##？？？
	bool consider_cluster_deletion;
	//num_cpus = None;
	Params1();
	~Params1();
};
#endif