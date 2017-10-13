#ifndef PARAMS__H__
#define  PARAMS__H__

#include <math.h>
#define STRLEN 10000

extern char FileBase[STRLEN];
extern  int ElecNo;
extern char UseFeatures[STRLEN];
extern int DropLastNFeatures;
extern int MaskStarts;
extern int MaxPossibleClusters;
extern int nStarts;
extern int UseDistributional;
extern int SplitEvery;
extern char StartCluFile[STRLEN];
extern int SplitFirst;
extern int MinClusters;
extern int MaxClusters;

extern float PenaltyK;
extern float PenaltyKLogN;
extern int Subset;
extern int FullStepEvery;

extern int MaxIter;
extern int RandomSeed;
extern int Debug;
extern int SplitInfo;
extern int Verbose;
extern int DistDump;
extern float DistThresh;
extern int Log;
extern int SaveTempCluEveryIter;

extern int Screen;
extern int PriorPoint;
extern int SaveSorted;
extern int SaveCovarianceMeans;
extern int UseMaskedInitialConditions;
extern int AssignToFirstClosestMask;
extern float RamLimitGB;
extern int  AlwaysSplitBimodal;
extern float PointsForClusterMask;
extern float MinMaskOverlap;
extern float ChangedThresh;

const float HugeScore = (float)1e30;


#endif