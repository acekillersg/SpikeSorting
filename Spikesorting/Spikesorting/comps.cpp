#pragma once
#pragma execution_character_set("utf-8")

#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <vector>
#include <utility>
#include <map>

using namespace std;


/*
该函数通过给定的阈值数组crossing（crossing数组中，如果某店高于高阈值，则值为2，若高于低阈值，则值为1，低于低阈值，值为0）,
通过flood_fill算法，进行测试提取的components。
*/
int main()
{
	vector < pair<int, int> > mark;
	pair<int, int> temp;
	vector < pair<int, int> > comp;
	map <int, vector < pair<int, int> >  > comps;

	int crossing[40][32];
	int strong_nodes[40][32];
	int strong_label[4*32];
	int label[40][32];
	memset(label, 0, sizeof(label));
	/****************************************/
	for (int i = 0; i < 40; i++)
		for (int j = 0; j < 32; j++){
			//if (i%2 == 0)
			crossing[i][j] = rand() % 3;
			printf("%d ", crossing[i][j]);
			//if (crossing[i][j] == 1)
			//printf("%d %d ", i,j);
			if (j == 31) printf("\n");
		}
	/******************************************/
	for (int i = 0; i < 40; i++)
		for (int j = 0; j<32; j ++)
		{
			if (crossing[i][j]>0){
			    temp = make_pair(i, j);
				mark.push_back(temp);
			}
			if (crossing[i][j] > 1)
				strong_nodes[i][j] = 1;
		}
	for (int i = 0; i < mark.size(); i++)
	{
		temp = mark[i];
		//printf("%d %d ", temp.first, temp.second);
	}
	/***********************************flood_fill algorithm*************************************************************/
	int c_label = 1;
	int join_size = 1;
	printf("------> weak crossing numbers: %d\n", mark.size());
	for (int i_s_ch = 0; i_s_ch < mark.size(); i_s_ch++){
		temp = mark[i_s_ch];
		int i_s = temp.first;
		int i_ch = temp.second;
		for (int j_s = max(i_s - join_size, 0);  j_s <= i_s;  j_s++){
			for (int j_ch = max(i_ch -  2, 0);   j_ch<= min(i_ch + 2,31);  j_ch++){
				int adjlabel = label[j_s][ j_ch];
				//printf(" j_s, j_ch, adjlabel---->:%d  %d  %d  \n", j_s, j_ch, adjlabel);
				if (adjlabel){
					int curlabel = label[i_s][i_ch];
					if (curlabel == 0){
						label[i_s][i_ch] = adjlabel;
						temp = make_pair(i_s, i_ch);
						comps[adjlabel].push_back(temp);
					}
					else if (curlabel != adjlabel){
						comp = comps[adjlabel];
						for (int k = 0; k < comp.size(); k++){
							temp = comp[k];
							comps[curlabel].push_back(temp);
							if (strong_label[adjlabel]){
								strong_label[adjlabel] = 0;
								strong_label[curlabel] = 1;
							}
						}
						comps.erase(adjlabel);
					}
					if (curlabel > 0 && crossing[i_s][i_ch]>1)
						strong_label[curlabel] = 1;
				}
			}
		}
		if (label[i_s][ i_ch] == 0){
			label[i_s][i_ch] = c_label;
			temp = make_pair(i_s, i_ch);
			comps[c_label].push_back(temp);
			if (crossing[i_s][i_ch] == 2)
				strong_label[c_label] = 1;
			c_label += 1;
		}
	}
/************************************************************************************************/
	printf("ssss\n");
	printf("%d\n", comps.size());
	vector < pair<int, int> > tmp;
	for (const auto &w : comps){
		tmp = w.second;
		for (int ii = 0; ii < tmp.size(); ii++)
			printf("[ %d %d ] ", tmp[ii].first, tmp[ii].second);
		printf("\n");
	}
	printf("ending...\n");
	getchar();
}