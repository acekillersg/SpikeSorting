#pragma once
#pragma execution_character_set("utf-8")

#include "probe.h"

Probe::Probe()
{
	/*---------------------------32channels probe's definition---------------------------*/
	//总的信道数（包含死亡信道）
	n_channels = 32;
	//信道的邻接边
	pair<int, int> t;
	for (int i = 0; i < 31; i++)
	{
		if (i + 2 < 32)
		{
			t = make_pair(i, i + 1);
			graph.push_back(t);
			t = make_pair(i, i + 2);
			graph.push_back(t);
		}
		else if (i + 1 < 32)
		{
			t = make_pair(i, i + 1);
			graph.push_back(t);
		}
	}
	//信道的物理位置
	int x;
	t = make_pair(0, 0);
	geometry[31] = t;
	for (int i = 30, j = 5, k = 10; i >= 0; i--, j++, k += 10){
		if (i % 2 == 0) x = j;
		else x = -j;
		t = make_pair(x, k);
		geometry[i] = t;
	}
	/*geometry = {
	{ 31, t }, { 32, t } };
	{ 31, make_pair(0, 0) },
	{ 30, make_pair(5, 10) },
	{29, make_pair(-6, 20)},
	{ 28, make_pair(7, 30) },
	{ 27, make_pair(-8, 40) },
	{ 26, make_pair(9, 50) },
	{ 25, make_pair(-10, 60) },
	{ 24, make_pair(11, 70) },
	{ 23, make_pair(-12, 80) },
	{ 22, make_pair(13, 90) },
	{ 21, make_pair(-14, 100) },
	{ 20, make_pair(15, 110) },
	{ 19, make_pair(-16, 120) },
	{ 18, make_pair(17, 130) },
	{ 17, make_pair(-18, 140) },
	{ 16, make_pair(19, 150) },
	{ 15, make_pair(-20, 160) },
	{ 14, make_pair(21, 170) },
	{ 13, make_pair(-22, 180) },
	{ 12, make_pair(23, 190) },
	{ 11, make_pair(-24, 200) },
	{ 10, make_pair(25, 210) },
	{ 9, make_pair(-26, 220) },
	{ 8, make_pair(27, 230) },
	{ 7, make_pair(-28, 240) },
	{ 6, make_pair(29, 250) },
	{ 5, make_pair(-30, 260) },
	{ 4, make_pair(31, 270) },
	{ 3, make_pair(-32, 280) },
	{ 2, make_pair(33, 290) },
	{ 1, make_pair(-34, 300) },
	{ 0, make_pair(35, 310) }};*/

}
Probe::~Probe(){}

//计算探针的邻接表
void Probe::edges_to_adjacency_list(map <int, vector < int >  > &probe_adjacency_list){
	pair <int, int > t;
	int x, y;
	vector<int> temp1;
	//每个信道自连接
	//for (int i = 0; i < n_channels; i++)
	//{
	//	vector<int> temp;
	//	temp.push_back(i);
	//	probe_adjacency_list[i] = temp;
	//}

	//每个信道的邻接表
	for (unsigned int i = 0; i < graph.size(); i++)
	{
		t = graph[i];
		x = t.first;
		y = t.second;

		temp1 = probe_adjacency_list[x];
		temp1.push_back(y);
		//probe_adjacency_list.erase(x);
		probe_adjacency_list[x] = temp1;

		temp1 = probe_adjacency_list[y];
		temp1.push_back(x);
		//probe_adjacency_list.erase(y);
		probe_adjacency_list[y] = temp1;
	}

	//测试邻接表输出
	//for (int i = 0; i < n_channels; i++){
	//	cout << i <<" : "<< endl;
	//	vector<int> te = probe_adjacency_list[i];
	//	for (int j = 0; j < te.size(); j++){
	//		cout << te[j] << " ";
	//	}
	//	cout << endl;
	//}

}

void Probe::set_graph(vector < pair<int, int> > &d_graph){
	for (int i = 0; i < d_graph.size(); i++){
		graph[i] = d_graph[i];
	}
}

void Probe::set_geometry(map <int, pair<int, int> > &d_geometry){
	for (int i = 0; i < n_channels; i++){
		d_geometry[i] = d_geometry[i];
	}
}

//测试探针的邻接图和探针物理位置输出
void Probe::test_output(){
	//邻接图
	for (int i = 0; i < graph.size(); i++)
		cout << "[" << graph[i].first << " " << graph[i].second << "]" << endl;
	//物理位置
	for (int i = 0; i < n_channels; i++)
		cout << "[" << geometry[i].first << " " << geometry[i].second << "]" << endl;
}

void Probe::dead_channels(vector<int > &dead_ch, int channels){
	for (int i = 0; i < graph.size(); i++){
		dead_ch[graph[i].first] = 1;
		dead_ch[graph[i].second] = 1;
	}
}