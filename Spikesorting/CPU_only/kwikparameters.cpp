#pragma once
#pragma execution_character_set("utf-8")
#include"kwikparameters.h"

Params1::Params1(){
	prior_point = 1; //？？？
	mua_point = 2;
	noise_point = 1;
	points_for_cluster_mask = 100; //？？？
	penalty_k = 0.0; //用于计算penalty
	penalty_k_log_n = 1.0; //用于计算penalty
	max_iterations = 1000;
	num_starting_clusters = 500;
	use_noise_cluster = true;
	use_mua_cluster = true;
	num_changed_threshold = 0.05; //？？？
	full_step_every = 1; //？？？
	split_first = 20;
	split_every = 40;
	MaxPossibleClusters = 1000;
	dist_thresh = log(10000.0); //？？？
	max_quick_step_candidates = 100000000; //this uses around 760 MB RAM  ##？？？
	max_quick_step_candidates_fraction = 0.4; //？？？
	always_split_bimodal = false; //？？？
	subset_break_fraction = 0.01; //？？？
	break_fraction = 0.0; //？？？
	fast_split = false; //？？？
	//max_split_iterations = None, ##？？？
	consider_cluster_deletion = true;
	//num_cpus = None;
}

Params1::~Params1(){}