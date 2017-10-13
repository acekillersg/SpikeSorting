#include<iostream>
#include<string>
#include<fstream>
#include<algorithm>
#include<cstdlib>
#include <vector>
#include <utility>
#include <map>
#include "spikedetect.h"
#include "probe.h"
#include "Eigen/Dense"
using namespace std;
using namespace Eigen;

//Ĭ�Ϲ��캯��
SpikeDetect::SpikeDetect(){
	for (int i = 0; i < samples; ++i)
		for (int j = 0; j < channels; ++j){
			traces[i][j] = 0.0;
			traces_t[i][j] = 0.0;
			crossing[i][j] = 0;
		}

	high_crossing = low_crossing = 0;
}

//���ı��ļ��й����������
SpikeDetect::SpikeDetect(std::string  filename){
	std::ifstream fin(filename);
	float in[samples];
	for (int i = 0; i < samples; ++i)
		for (int j = 0; j < channels; ++j){
			fin >> in[j];
			traces[i][j] = in[j];
			crossing[i][j] = 0;
			traces_t[i][j] = in[j];
		}
	//һЩ�м����ĳ�ʼ��
	high_crossing = low_crossing = 0;
}

//��������
SpikeDetect :: ~SpikeDetect(){
}

//�������
void SpikeDetect::output(){

	printf("%f\n", traces_t[814][13]);
	//for (int i = 0; i < 10; i++){
	//	for (int j = 0; j < channels; j++)
	//		cout << traces[i][j] << "  ";
	//	cout << endl;
	//}
	//cout << "========" << endl;
	//for (int i = samples - 10; i < samples; i++){
	//	for (int j = 0; j < channels; j++)
	//		cout << traces[i][j] << "   ";
	//	cout << endl;
	//}
	cout << high_crossing << " " << low_crossing << endl;
}

//�˲�����
void SpikeDetect::filter(int rate, float high, float low, int order){
}

//������ֵ
void SpikeDetect::threshold(){
	float median = 0.0;
	/*-------------------------����λ������Ҫ�Ľ�-----------------*/
	int nums = samples * channels;
	float temp[samples * channels];
	int x = 0;
	for (int i = 0; i < samples; i++)
		for (int j = 0; j < channels; j++)
		{
			if (traces[i][j] < 0)
				temp[x++] = -traces[i][j];
			else
				temp[x++] = traces[i][j];
		}
	std::stable_sort(temp, temp + nums);
	/*----------------------------------------------------------------*/
	if (nums % 2) median = temp[nums / 2];
	else median = (float)((temp[nums / 2] + temp[nums / 2 + 1]) / 2.0);
	high_crossing = (float)(p.get_threshold_strong_std_factor() * median / 0.6745);
	low_crossing =(float)(p.get_threshold_weak_std_factor() * median / 0.6745);
}

//�����ν��з�ת����
void SpikeDetect::transform(){
	if (p.get_detect_spikes_symbol() == -1){
		cout << "detect_spikes_symbol  ==  -1" << endl;
		for (int i = 0; i < samples; i++)
			for (int j = 0; j < channels; j++)
				traces_t[i][j] = -traces[i][j];
	}
	else if (p.get_detect_spikes_symbol() == 1){
		for (int i = 0; i < samples; i++)
			for (int j = 0; j < channels; j++)
				if (traces[i][j] < 0)
					traces_t[i][j] = -traces[i][j];
	}
}

//����componments
void SpikeDetect::detect(map <int, vector < pair<int, int> >  > &comps){

	map <int, vector < pair<int, int> >  > comps_temp;
	std::vector < std::pair<int, int> > mark;
	pair<int, int> temp;
	vector < pair<int, int> > comp;

	int strong_nodes[samples][channels];
	int strong_label[samples * channels];
	int label[samples][channels];
	memset(label, 0, sizeof(label));
	memset(strong_nodes, 0, sizeof(strong_nodes));
	/****************************************/
	for (int i = 0; i < samples; i++){
		//printf("line: %d  ::", i);
		for (int j = 0; j < channels; j++)
		{
			if (traces_t[i][j] >= high_crossing)
				crossing[i][j] = 2;
			else if (traces_t[i][j] >= low_crossing)
				crossing[i][j] = 1;
			//printf("%d  ", crossing[i][j]);
		}
		//printf("\n");
	}
		
	/******************************************/
	for (int i = 0; i < samples; i++)
		for (int j = 0; j<channels; j++)
		{
			
			if (crossing[i][j]>0){
				temp = make_pair(i, j);
				mark.push_back(temp);
			}
			if (crossing[i][j] == 2)
				strong_nodes[i][j] = 1;
		}
	/***********************************flood_fill algorithm*************************************************************/
	int c_label = 1;
	int join_size = p.get_connected_component_join_size();
	//printf("------> weak crossing numbers: %d           %d\n", mark.size(),scnt);
	for (unsigned int i_s_ch = 0; i_s_ch < mark.size(); i_s_ch++){
		temp = mark[i_s_ch];
		int i_s = temp.first;
		int i_ch = temp.second;
		for (int j_s = max(i_s - join_size, 0); j_s <= i_s; j_s++){
			for (int j_ch = max(i_ch - 2, 0); j_ch <= min(i_ch + 2, 31); j_ch++){
				int adjlabel = label[j_s][j_ch];
				//printf(" j_s, j_ch, adjlabel---->:%d  %d  %d  \n", j_s, j_ch, adjlabel);
				if (adjlabel!=0){
					int curlabel = label[i_s][i_ch];
					if (curlabel == 0){
						label[i_s][i_ch] = adjlabel;
						temp = make_pair(i_s, i_ch);
						comps_temp[adjlabel].push_back(temp);

						if (crossing[i_s][i_ch]==2)
							strong_label[adjlabel] = 1;
					}
					else if (curlabel != adjlabel){
						comp = comps_temp[adjlabel];
						for (unsigned int k = 0; k < comp.size(); k++){
							temp = comp[k];
							label[temp.first][temp.second] = curlabel;
							comps_temp[curlabel].push_back(temp);
							}
						if (strong_label[adjlabel] == 1 ||  crossing[i_s][i_ch]>1)
								strong_label[adjlabel] = 0,strong_label[curlabel] = 1;
						comps_temp.erase(adjlabel);
					}
					else{
						if (crossing[i_s][i_ch] == 2) 
							strong_label[adjlabel] = 1, strong_label[curlabel] = 1;
					}
				}
			}
		}
		if (label[i_s][i_ch] == 0){
			label[i_s][i_ch] = c_label;
			temp = make_pair(i_s, i_ch);
			comps_temp[c_label].push_back(temp);
			if (crossing[i_s][i_ch] == 2)
				strong_label[c_label] = 1;
			c_label += 1;
		}
	}
	//��û�и���ֵ���compȥ��
	vector < pair<int, int> > tmp;
	int sp = 0;
	for (const auto &w : comps_temp){
		int cc = w.first;
		if (strong_label[cc] == 1){
			sp++;
			tmp = w.second;
			comps[sp] = tmp;
		}
	}
	/******************************************test*************************************************/
	/*sp = 1;
	for (const auto &w : comps){
		int cc = w.first;
		printf("this is the %d spike's component", sp++);
		tmp = w.second;
		for (unsigned int ii = 0; ii < tmp.size(); ii++)
	        printf("[ %d %d ] ", tmp[ii].first, tmp[ii].second);
		 printf("\n\n");
	}
	printf("this is all %d spike,ending...\n",sp-1);
	getchar();*/

}

void SpikeDetect::flood_fill(int x, int y, int label)
{
	crossing[x][y] = label;
	if (x > 0 && crossing[x - 1][y] == 0)
		flood_fill(x - 1, y, label);
	if (y > 0 && crossing[x][y - 1] == 0)
		flood_fill(x, y - 1, label);
	if (y > 0 && crossing[x][y - 1] == 0)
		flood_fill(x, y - 1, label);
	if (x< samples && crossing[x + 1][y] == 0)
		flood_fill(x + 1, y, label);
	if (y< channels && crossing[x][y + 1] == 0)
		flood_fill(x, y + 1, label);
}

//void SpikeDetect::detect(map <int, vector < pair<int, int> >  > &comps,vector<int> ){
//	//strong_node�����и��ڸ���ֵ�ĵ�ļ���
//	vector <pair <int, int>> strong_node;
//	pair <int, int > temp;
//	//���������ŵ�
//	//crossing
//	for (int i = 0; i < samples; i++)
//		for (int j = 0; j < channels; j++)
//		{
//			if (traces[i][j] > high_crossing)
//			{
//				crossing[i][j] = 2;
//				temp = make_pair(i, j);
//				strong_node.push_back(temp);
//			}
//			else if (traces[i][j] > low_crossing)
//				crossing[i][j] = 1;
//		}
//}

////���ݸߵ���ֵ��һ�����źŵĵ�λֵ
float SpikeDetect::normalize(float x){
	float norm_x;
	norm_x = (x - low_crossing) / (high_crossing - low_crossing);
	if (norm_x > 1) norm_x = 1;
	if (norm_x < 0) norm_x = 0;
	return norm_x;
}

////����componment��traces_t����ȡwave,
void SpikeDetect::comp_wave(int s_min, int s_max, vector < pair<int, int> > tmp_comp, std::vector<std::vector <float> > &wave){

	//memset(wave, 0, sizeof(wave));
	for (unsigned int i = 0; i < tmp_comp.size(); i++){
		int x = tmp_comp[i].first;
		int y = tmp_comp[i].second;
		wave[x - s_min][y] = traces_t[x][y];
	}
	/*---------------test---------------*/
	/*for (int i = 0; i <= s_max - s_min; i++){
		for (int j = 0; j < channels; j++)
			cout << wave[i][j] << " ";
		cout << endl<< endl;
	}*/
}

////����wave��componment������mask
void SpikeDetect::mask(int s_min, int s_max, vector<float > &mask, vector<int > &mask_bin, std::vector<std::vector <float> > &wave){
	for (unsigned int i = 0; i < mask_bin.size(); i++){
		if (mask_bin[i]==1){
			float peak = wave[0][i];
			for (int j = 1; j <= s_max-s_min; j++)
				if (peak < wave[j][i])
					peak = wave[j][i];
			mask[i] = normalize(peak);
		}//if
	}//for
	/*---------------test---------------*/
	/*for (int i = 0; i < mask.size(); i++)
		cout << mask[i] << " ";
	cout << endl;*/
}

////����ÿ��wave������ʱ��,���м������ʱ����GT��jetter<=12,����Ϊ���ζ��ճɹ�
void SpikeDetect::spike_sample_aligned(int s_min, int s_max, float &s_aligned, std::vector<std::vector <float> > &wave){

	float sum_t1 = 0.0, sum_t2 = 0.0;
	for (int i = 0; i <= s_max - s_min; i++)
		for (int j = 0; j < channels; j++){
			float wave_n, wave_n_p;
			wave_n = normalize(wave[i][j]);
			wave_n_p = wave_n*wave_n;
			sum_t1 += (wave_n_p)*i;
			sum_t2 += wave_n_p;
		}
	s_aligned = sum_t1 / sum_t2 + (float)s_min;

	/*---------------test---------------*/
	//cout << s_aligned << endl;
}

////����wave���Ķ���ʱ�䣬��ȡwaveform(Ҫ�ڴ˴�ȥ�������ŵ�)
void SpikeDetect::extract(int before, int after, float &s_aligned, std::vector <std::vector <float> > &waveform){
	int sa = (int)s_aligned;
	int s_start = sa - before - 1;
	int s_end = sa + after + 2;
	if (s_start < 0 && s_end >= samples){
		cout << "we think it's not  a wavrform!!!" << endl;
		return;
	}
	//tracesΪFilter��Ĳ���
	else if (s_start < 0 || s_end >= samples){
		for (int i = 0; i < s_end - s_start; i++)
			for (int j = 0; j < channels; j++){
				if (i + s_start < 0 || i+s_start >= samples) waveform[i][j] = 0.0;
				else waveform[i][j] = traces[i + s_start][j];
			}
	}
	else{
		for (int i = 0; i < s_end - s_start; i++)
			for (int j = 0; j < channels; j++)
				waveform[i][j] = traces[i + s_start][j];
	}
	/*---------------test---------------*/
	/*for (int i = 0; i < before + after + 3; i++)
	{
		for (int j = 0; j < channels; j++)
			cout << waveform[i][j] << "  ";
		cout << endl;
	}*/
}

////����������ֵʹ����ȡ�Ĳ���ƽ����
void SpikeDetect::align(float s_aligned, int before, int after, std::vector <std::vector <float> > &waveform, std::vector <std::vector <float> > &wavealign)
{
	int s_align = (int)s_aligned;
	int width = before + after + 3;
	int start = s_align - before - 1;
	float offset = s_aligned - (float)s_align+start + 1;

	Vector<float> x(width),y(width);
	for (int i = 0; i < channels; i++){
		for (int j = 0; j < width; j++){
			x[j] = (float)(j + start);
			y[j] = waveform[j][i];
		}
		int Ml = 0, Mr = 0;
	    Spline3Interp<float> poly(x, y, Ml, Mr);
		//����ÿ�ε��ĸ�ϵ����������poly.getCoefs()�õ����е�ϵ��
		poly.calcCoefs();
		//�������ǲ������ƽ�����ֵ
		for (int j = 0; j < before + after; j++)
			wavealign[j][i] = poly.evaluate((float)j +offset);
		/*---------------test---------------*/
		/*for (int j = 0; j < before + after; j++)
			cout << wavealign[j][i] << "  ";
		cout << endl<<endl;*/
	}
}
	
/*--------------------compute PCA--------------------------*/
void featurenormalize(MatrixXf &X)
{
	//����ÿһά�Ⱦ�ֵ
	MatrixXf meanval = X.colwise().mean();
	RowVectorXf meanvecRow = meanval;
	//������ֵ��Ϊ0
	X.rowwise() -= meanvecRow;
}

void computeCov(MatrixXf &X, MatrixXf &C)
{
	//����Э�������C = XTX /( n-1);
	C = X.adjoint() * X;
	C = C.array() / (X.rows() - 1);
}

void computeEig(MatrixXf &C, MatrixXf &vec, MatrixXf &val){
	//��������ֵ������������ʹ��selfadjont���ն��������㷨ȥ���㣬�����ò�����vec��val������������
	SelfAdjointEigenSolver<MatrixXf> eig(C);
	vec = eig.eigenvectors();
	val = eig.eigenvalues();

	//EigenSolver<MatrixXf> eig(C);
    //val = eig.pseudoEigenvalueMatrix();
	//vec = eig.pseudoEigenvectors();
	//cout << "Finally, V * D * V^(-1) = " << endl << vec * val * vec.inverse() << endl;
}

int computeDim(MatrixXf &val)
{
	int dim;
	double sum = 0;
	for (int i = val.rows() - 1; i >= 0; --i)
	{
		sum += val(i, 0);
		dim = i;

		if (sum / val.sum() >= 0.95)
			break;
	}
	return val.rows() - dim;
}
/*--------------------------------------------------------------*/
////����waveforms�����ɷ�
void SpikeDetect::PCA(int n_spikes, std::vector< std::vector< std::vector <float> > >&waveforms, std::vector <std::vector <float> > &masks, std::vector< std::vector< std::vector <float> > >&out){
	int width_sample = p.get_extract_s_after() + p.get_extract_s_before();
	int n_pca = p.get_n_features_per_channel();

	//��������δ���ڱ�������PCA֮���������pcs��������Щ������ÿ��channelһ��width_sample*n_pca��������pcs.shape= (n_pca , width_sample, n_channels)
	std::vector< std::vector< std::vector <float> > >pcs(n_pca);
	for (int i = 0; i < n_pca; i++){
		pcs[i].resize(width_sample);
		for (int j = 0; j < width_sample; j++)
			pcs[i][j].resize(channels);
	}
	
	for (int i = 0; i < channels; i++)
	{
		//cout << "this is  " << i << " channel" << endl;
		int n_spike_ch = 0;
		vector <int> mk;
		for (int j = 0; j < n_spikes; j++)
			if (masks[j][i]> 0){
				n_spike_ch++;
				mk.push_back(j);
			}
				

		Eigen::MatrixXf cov_reg(width_sample, width_sample);
		Eigen::MatrixXf cov(width_sample, width_sample);

		for (int ii = 0; ii < width_sample; ii++){
			for (int jj = 0; jj < width_sample; jj++){
			    if (ii == jj ) cov_reg(ii, jj) = 1.0;
			    else cov_reg(ii, jj) = 0;
				cov(ii, jj) = 0;
		    }
		}
			
		if (n_spike_ch <= 1)
			cov = (1.0 / n_spikes) * cov_reg;
		else
		{
			Eigen::MatrixXf X(n_spike_ch, width_sample);
			for (int j = 0; j < n_spike_ch; j++)
					for (int k = 0; k < width_sample; k++)
						X(j, k) = waveforms[ mk[j] ] [k] [i];

			/*for (int j = 0; j < n_spike_ch; j++){
				for (int k = 0; k < width_sample; k++)
					cout << X(j, k) << "  ";
				cout << endl;
			}*/

			Eigen::MatrixXf cov_channel(width_sample, width_sample);
			featurenormalize(X);
			computeCov(X, cov_channel);
			cov = (1.0 / n_spikes) * cov_reg + cov_channel;
		}

		MatrixXf vec, val;
		computeEig(cov, vec, val);

		////ȡÿ��channel�����������ϵ�ǰn_pcaά���pcs
		for (int np = 0; np < n_pca; np++)
			for (int ns = 0; ns < width_sample; ns++)
				pcs[np][ns][i] = vec(ns, width_sample - 1 - np);
	}
	/*for (int i = 0; i < n_pca; i++)
	{
		for (int j = 0; j < width_sample; j++)
		{
			for (int k = 0; k < channels; k++)
				cout << pcs[i][j][k] << "  ";
			cout << endl;
		}
		cout << endl;
	}*/

	//ͨ��pcs��x����x���н�ά
	for (int nsp = 0; nsp < n_spikes; nsp++)
		for (int nf = 0; nf < n_pca; nf++)
			for (int ch = 0; ch < channels; ch++){
				float sum = 0.0;
				for (int ns = 0; ns < width_sample; ns++)
					sum += pcs[nf][ns][ch] *waveforms[nsp][ns][ch];
				out[nsp][ch][nf] = sum;
			}

	/*---------------test---------------*/
	/*for (int i = 0; i < n_spikes; i++)
	{
		for (int j = 0; j < channels; j++)
		{
			for (int k = 0; k < n_pca; k++)
				cout << out[i][j][k] << "  ";
			cout << endl;
		}
		cout << endl;
	}*/
}