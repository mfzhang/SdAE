#ifndef _PARAM_
#define _PARAM_

#include <iostream>
#include <vector>

#include <stdlib.h>
#include <string.h>

using namespace std;

class Param
{
public:
	double 	Alpha;
	
	double 	Dropout_rate;
	int 		Dropout_time;
	
	int 		dAE_max_iteration;
	int		BP_max_iteration;
	int		LR_max_iteration;
	
	double	dAE_stop_threshold;
	double	BP_stop_threshold;
	double	LR_stop_threshold;
	
	vector<int>	Node_Sizes;
	int			Layer_Size;
	
	Param();
	Param(int argc,char* argv[]);
	void dump();
};

#endif
