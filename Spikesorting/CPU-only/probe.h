//#pragma once
//#pragma execution_character_set("utf-8")

#include <iostream>
#include <utility>
#include <vector>
#include <map>
#include <set>
#ifndef PROBE__H__
#define PROBE__H__

using namespace  std;

class Probe{

private:
	int n_channels;
	//static const int num = 2500;
	vector < pair <int, int> > graph;
	map < int, pair<int, int> > geometry;

public:
	Probe();
	~Probe();
	void edges_to_adjacency_list(std::map <int, vector < int >  > &comps);
	void set_graph(vector < pair<int, int> > &d_graph);
	void set_geometry(map <int, pair<int, int> > &d_geometry);
	void dead_channels(vector<int > &dead_ch, int channels);
	void test_output();
	//void get_graph();
	//void get_geometry();
};
#endif