#ifndef _DATASET_
#define _DATASET_

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>

#include <stdio.h>
#include <math.h>
#include <dirent.h>

#include "Utility.h"

#include <Eigen/Eigen>

using namespace std;
using namespace Eigen;

class DataSet
{
public:
	std::vector<VectorXd>	X;
	std::vector<VectorXd>	Y;
	
	int		data_size;
	int		feature_size;
	int		label_size;
	
private:
	void init_Feature_Size(std::string );
	void init_Label_Size(std::string );
	
	void load_X(std::string );
	void load_Y(std::string );

public:
		DataSet(std::string );
	void set(std::string );
	void normalize();
};

#endif
