#include "params.h"

char FileBase[STRLEN] = "electrode";
int ElecNo = 1;
char UseFeatures[STRLEN] = "";
int DropLastNFeatures = 0;
int MaskStarts = 500;
int MaxPossibleClusters = 1000;
int nStarts = 1;
int UseDistributional = 1;
int SplitEvery = 40;
char StartCluFile[STRLEN] = "";
int SplitFirst = 20;
int MinClusters = 100;
int MaxClusters = 110;

float PenaltyK = 0.0;
float PenaltyKLogN = 1.0;
int Subset = 1;
int FullStepEvery = 20;

int MaxIter = 10000;
int RandomSeed = 1;
int Debug = 0;
int SplitInfo = 1;
int Verbose = 1;
int DistDump = 0;
float DistThresh = (float)log(1000.0);
int Log = 1;
int SaveTempCluEveryIter = 0;

int Screen = 1;
int PriorPoint = 1;
int SaveSorted = 0;
int SaveCovarianceMeans = 0;
int UseMaskedInitialConditions = 0;
int AssignToFirstClosestMask = 0;
float RamLimitGB = 0.0;
int  AlwaysSplitBimodal = 0;
float PointsForClusterMask = 10.0;
float MinMaskOverlap = 0.0;

float ChangedThresh = 0.05;