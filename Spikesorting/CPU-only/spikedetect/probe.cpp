#pragma once
#pragma execution_character_set("utf-8")

#include "probe.h"

Probe::Probe()
{
	/*---------------------------32channels probe's definition---------------------------*/
	//�ܵ��ŵ��������������ŵ���
	n_channels = 32;
	//�ŵ����ڽӱ�
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
	//�ŵ�������λ��
	int x;
	t = make_pair(0, 0);
	geometry[31] = t;
	for (int i = 30, j = 5, k = 10; i >= 0; i--, j++, k += 10){
		if (i % 2 == 0) x = j;
		else x = -j;
		t = make_pair(x, k);
		geometry[i] = t;
	}
}
Probe::~Probe(){}

//����̽����ڽӱ�
void Probe::edges_to_adjacency_list(map <int, vector < int >  > &probe_adjacency_list){
	pair <int, int > t;
	int x, y;
	vector<int> temp1;
	//ÿ���ŵ�������
	//for (int i = 0; i < n_channels; i++)
	//{
	//	vector<int> temp;
	//	temp.push_back(i);
	//	probe_adjacency_list[i] = temp;
	//}

	//ÿ���ŵ����ڽӱ�
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

	//�����ڽӱ����
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
	for (unsigned int i = 0; i < d_graph.size(); i++){
		graph[i] = d_graph[i];
	}
}

void Probe::set_geometry(map <int, pair<int, int> > &d_geometry){
	for (int i = 0; i < n_channels; i++){
		geometry[i] = d_geometry[i];
	}
}

//����̽����ڽ�ͼ��̽������λ�����
void Probe::test_output(){
	//�ڽ�ͼ
	for (unsigned int i = 0; i < graph.size(); i++)
		cout << "[" << graph[i].first << " " << graph[i].second << "]" << endl;
	//����λ��
	for (int i = 0; i < n_channels; i++)
		cout << "[" << geometry[i].first << " " << geometry[i].second << "]" << endl;
}

void Probe::dead_channels(vector<int > &dead_ch, int channels){
	for (unsigned int i = 0; i < graph.size(); i++){
		dead_ch[graph[i].first] = 1;
		dead_ch[graph[i].second] = 1;
	}
}