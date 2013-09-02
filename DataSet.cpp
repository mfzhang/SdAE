#include "DataSet.h"

void DataSet::init_Sizes(string target_path)
{
	feature_size	= -1;
	label_size	= -1;
	ifstream ifs(	(target_path).c_str() );
	for(string line ; getline(ifs, line) ;)
	{
		vector<string> each_term = uStr::split(line,' ');
		for(int j=1;j<each_term.size() ;j++)
			feature_size  = max(feature_size, atoi(uStr::split(each_term[j],':')[0].c_str())	);
		label_size  = max(label_size, atoi(each_term[0].c_str()));
	}
	feature_size = feature_size + 2;
	label_size += 1;
}


void DataSet::load(string target_path)
{
	X =	vector<VectorXd>();
	Y =	vector<VectorXd>();

	ifstream ifs( (target_path).c_str());
	int feature_index;
	
	VectorXd tmp_x;
	VectorXd tmp_y;
	for(string line ; getline(ifs, line);)
	{
		tmp_x = VectorXd::Zero(feature_size);
		tmp_y = VectorXd::Zero(label_size);
		vector<string> each_term = uStr::split(line,' ');
		for(int i=1;i<each_term.size();i++ )
		{
			vector<string> fact = uStr::split(each_term[i] , ':');
			feature_index = atoi(fact[0].c_str());
			if(feature_index < feature_size)
				tmp_x(feature_index) = atof(fact[1].c_str());
		}
		tmp_x(feature_size-1) = 1;
		X.push_back(tmp_x);
		
		tmp_y(atoi(each_term[0].c_str())) = 1;
		Y.push_back(tmp_y);
	}
	ifs.close();
}

DataSet::DataSet(string target_dir)
{
	if(target_dir[target_dir.size()] != '/')
		target_dir += "/";
	init_Sizes(target_dir + "X");
}

void DataSet::set(string target_dir)
{
	if(target_dir[target_dir.size()] != '/')
		target_dir += "/";
	load(target_dir + "X");
}

void DataSet::normalize()
{
	#pragma omp parallel for
	for(int i=0;i<X.size();i++)
		X[i] /= X[i].sum();		
}
