// Disable some Visual Studio warnings
#define _CRT_SECURE_NO_WARNINGS

#include "klustakwik.h"

// Loads in Fet file.  Also allocates storage for other arrays
void KK::LoadData(char *FileBase, int ElecNo, char *UseFeatures)
{
	char fname[STRLEN];
	char fnamefmask[STRLEN];
	char line[STRLEN];
	int p, i, j;

	int nFeatures, nmaskFeatures;
	FILE *fp;
	FILE *fpfmask;
	int status;
	float val;
	int UseLen;

	// open file
	sprintf(fname, "%s.fet.%d", FileBase, (int)ElecNo);
	fp = fopen_safe(fname, "r");

	if ((MaskStarts > 0) && UseDistributional)
	{
		Output("-------------------------------------------------------------------------");
		Output("\nUsing Distributional EM with Maskstarts\n");
		MinClusters = MaskStarts;
		MaxClusters = MaskStarts;
		Output("NOTE: Maskstarts overides above values of MinClusters and MaxClusters \
			   			                   \nMinClusters = %d \nMaxClusters = %d \n ", (int)MinClusters, (int)MaxClusters);
	}

	sprintf(fnamefmask, "%s.fmask.%d", FileBase, (int)ElecNo);
	fpfmask = fopen_safe(fnamefmask, "r");


	// count lines;
	nPoints = -1; // subtract 1 because first line is number of features
	while (fgets(line, STRLEN, fp)) {
		nPoints++;
	}

	// rewind file
	fseek(fp, 0, SEEK_SET);

	// read in number of features
	fscanf(fp, "%d", &nFeatures);

	// calculate number of dimensions
	if (UseFeatures[0] == 0)
	{
		nDims = nFeatures - DropLastNFeatures; // Use all but the last N Features.
		UseLen = nFeatures - DropLastNFeatures;
	}
	else
	{
		UseLen = strlen(UseFeatures);
		nDims = 0;
		for (i = 0; i<nFeatures; i++)
		{
			nDims += (i<UseLen && UseFeatures[i] == '1');
		}
	}
	nDims2 = nDims*nDims;
	//MemoryCheck();
	AllocateArrays();

	// load data
	Output(" nPoints:  %d     nFeatures: %d\n", nPoints, nFeatures);
	for (p = 0; p<nPoints; p++) {
		j = 0;
		for (i = 0; i<nFeatures; i++) {
			float readfloatval;
			status = fscanf(fp, "%f", &readfloatval);
			val = (float)readfloatval;
			if (status == EOF) Output("Error reading feature file"), getchar();

			if (UseFeatures[0] == 0) //when we want all the features
			{
				if (i<UseLen)
					Data[p*nDims + j++] = val;
			}
			else  // When we want the subset specified by the binary string UseFeatures, e.g. 111000111010101
			{
				if (i<UseLen && UseFeatures[i] == '1')
					Data[p*nDims + j++] = val;
			}
		}
	}

	if (UseDistributional) //replaces if(UseFloatMasks)
	{
		// rewind file
		fseek(fpfmask, 0, SEEK_SET);

		// read in number of features
		fscanf(fpfmask, "%d", &nmaskFeatures);

		if (nFeatures != nmaskFeatures)
			Output("Error: Float Mask file and Fet file incompatible");

		// load float masks
		for (p = 0; p<nPoints; p++) {
			j = 0;
			for (i = 0; i<nFeatures; i++)
			{
				float readfloatval;
				status = fscanf(fpfmask, "%f", &readfloatval);
				if (status == EOF) Output("Error reading fmask file"), getchar();
				val = (float)readfloatval;

				if (UseFeatures[0] == 0)
				{
					if (i<UseLen)
					{
						FloatMasks[p*nDims + j] = val;
						j++;
					}
				}
				else  // When we want all the features
				{
					if (i<UseLen && UseFeatures[i] == '1') //To Do: implement DropLastNFeatures
					{
						FloatMasks[p*nDims + j] = val;
						j++;
					}
				}
			}
		}
	}

	if (UseDistributional)
	{
		for (p = 0; p<nPoints; p++)
			for (i = 0; i<nDims; i++)
			{
				if (FloatMasks[p*nDims + i] == (float)1) //changed so that this gives the connected component masks
					Masks[p*nDims + i] = 1;
				else
					Masks[p*nDims + i] = 0;
			}
	}
	else  //Case for Classical EM KlustaKwik
	{
		for (p = 0; p<nPoints; p++)
			for (i = 0; i<nDims; i++)
				Masks[p*nDims + i] = 1;
	}
	fclose(fp);
	if (UseDistributional)
		fclose(fpfmask);

	Output("----------------------------------------------------------\nLoaded %d data points of dimension %d.\n", (int)nPoints, (int)nDims);
	Output("MEMO: A lower score indicates a better clustering \n ");
}

// write output to .clu file - with 1 added to cluster numbers, and empties removed.
void KK::SaveOutput()
{
	int c;
	unsigned int p;
	char fname[STRLEN];
	FILE *fp;
	int MaxClass = 0;
	vector<int> NotEmpty(MaxPossibleClusters);
	vector<int> NewLabel(MaxPossibleClusters);

	// find non-empty clusters
	for (c = 0; c<MaxPossibleClusters; c++) NewLabel[c] = NotEmpty[c] = 0;
	for (p = 0; p<BestClass.size(); p++) NotEmpty[BestClass[p]] = 1;

	// make new cluster labels so we don't have empty ones
	NewLabel[0] = 1;
	MaxClass = 1;
	for (c = 1; c<MaxPossibleClusters; c++) {
		if (NotEmpty[c]) {
			MaxClass++;
			NewLabel[c] = MaxClass;
		}
	}

	// print file
	sprintf(fname, "%s.clu.%d", FileBase, (int)ElecNo);
	fp = fopen_safe(fname, "w");

	fprintf(fp, "%d\n", (int)MaxClass);
	for (p = 0; p<BestClass.size(); p++) fprintf(fp, "%d\n", (int)NewLabel[BestClass[p]]);

	fclose(fp);

	if (SaveCovarianceMeans)
		SaveCovMeans();
	if (SaveSorted&&UseDistributional)
		SaveSortedClu();
}

// write output to .clu file - with 1 added to cluster numbers, and empties removed.
void KK::SaveTempOutput()
{
	int c;
	unsigned int p;
	char fname[STRLEN];
	FILE *fp;

	int MaxClass = 0;

	vector<int> NotEmpty(MaxPossibleClusters);

	vector<int> NewLabel(MaxPossibleClusters);

	// find non-empty clusters
	for (c = 0; c<MaxPossibleClusters; c++) NewLabel[c] = NotEmpty[c] = 0;
	// We are merely storing the results of the current iteration, it may not be the best so far
	for (p = 0; p<Class.size(); p++) NotEmpty[Class[p]] = 1;

	// make new cluster labels so we don't have empty ones
	NewLabel[0] = 1;
	MaxClass = 1;
	for (c = 1; c<MaxPossibleClusters; c++) {
		if (NotEmpty[c]) {
			MaxClass++;
			NewLabel[c] = MaxClass;
		}
	}

	// print temp.clu file
	//This is the clu for the current iteration
	//This fixes the bug of having a trivial temp.clu file if there is only one iteration
	sprintf(fname, "%s.temp.clu.%d", FileBase, (int)ElecNo);
	fp = fopen_safe(fname, "w");

	fprintf(fp, "%d\n", (int)MaxClass);
	for (p = 0; p<Class.size(); p++) fprintf(fp, "%d\n", (int)NewLabel[Class[p]]);
	fclose(fp);

	if (SaveCovarianceMeans)
		SaveCovMeans();
	if (SaveSorted&&UseDistributional)
		SaveSortedClu();
}

void KK::SaveCovMeans()
{
	char fname[STRLEN];
	FILE *fp;
	// print covariance to file
	sprintf(fname, "%s.cov.%d", FileBase, (int)ElecNo);
	fp = fopen_safe(fname, "w");
	for (int cc = 0; cc<nClustersAlive; cc++)
	{
		int c = AliveIndex[cc];
		for (int i = 0; i<nDims; i++)
		{
			for (int j = 0; j<nDims; j++)
			{
				// TODO: update Cov output for distributional
				if (!UseDistributional)
					fprintf(fp, "%f ", Cov[c*nDims2 + i*nDims + j]);
			}
			fprintf(fp, "\n");
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
	// print mean to file
	sprintf(fname, "%s.mean.%d", FileBase, (int)ElecNo);
	fp = fopen_safe(fname, "w");
	for (int cc = 0; cc<nClustersAlive; cc++)
	{
		int c = AliveIndex[cc];
		for (int i = 0; i<nDims; i++)
		{
			fprintf(fp, "%f ", Mean[c*nDims + i]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
}

// Saves sorted.fet and sorted.mask file
void KK::SaveSortedData()
{
	char fname[STRLEN];
	FILE *fp;
	// sorted.fet file
	sprintf(fname, "%s.sorted.fet.%d", FileBase, (int)ElecNo);
	fp = fopen_safe(fname, "w");
	fprintf(fp, "%d\n", (int)nDims);
	for (int q = 0; q<nPoints; q++)
	{
		int p = SortedIndices[q];
		for (int i = 0; i<nDims; i++)
			fprintf(fp, "%f ", GetData(p, i));
		fprintf(fp, "\n");
	}
	fclose(fp);
	// sorted.mask file
	sprintf(fname, "%s.sorted.mask.%d", FileBase, (int)ElecNo);
	fp = fopen_safe(fname, "w");
	fprintf(fp, "%d\n", (int)nDims);
	for (int q = 0; q<nPoints; q++)
	{
		int p = SortedIndices[q];
		for (int i = 0; i<nDims; i++)
			fprintf(fp, "%d ", (int)GetMasks(p*nDims + i));
		fprintf(fp, "\n");
	}
	fclose(fp);
}

// Save sorted.clu file (see SaveOutput for explanation)
void KK::SaveSortedClu()
{
	char fname[STRLEN];
	FILE *fp;
	vector<int> NotEmpty(MaxPossibleClusters);
	vector<int> NewLabel(MaxPossibleClusters);
	for (int c = 0; c<MaxPossibleClusters; c++)
		NewLabel[c] = NotEmpty[c] = 0;
	for (int q = 0; q<nPoints; q++)
		NotEmpty[Class[SortedIndices[q]]] = 1;
	NewLabel[0] = 1;
	int MaxClass = 1;
	for (int c = 1; c<MaxPossibleClusters; c++)
		if (NotEmpty[c])
			NewLabel[c] = ++MaxClass;
	sprintf(fname, "%s.sorted.clu.%d", FileBase, (int)ElecNo);
	fp = fopen_safe(fname, "w");
	fprintf(fp, "%d\n", (int)MaxClass);
	for (int q = 0; q<nPoints; q++)
		fprintf(fp, "%d\n", (int)NewLabel[Class[SortedIndices[q]]]);
	fclose(fp);
}
