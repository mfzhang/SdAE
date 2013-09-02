#include "DataSet.h"

void DataSet::init_Feature_Size(string target_path)
{
	feature_size	= -1;
	ifstream ifs(	(target_path).c_str() );
	for(string line ; getline(ifs, line) ;)
	{
		vector<string> each_term = uStr::split(line,' ');
		for(int j=0;j<each_term.size() ;j++)
			feature_size  = max(feature_size, atoi(uStr::split(each_term[j],':')[0].c_str())	);
	}
	feature_size = feature_size + 1;
}

void DataSet::init_Label_Size(string target_path)
{
	label_size	= -1;
	ifstream ifs(	(target_path).c_str() );
	for(string line ; getline(ifs, line) ;)
	{
		vector<string> each_term = uStr::split(line,' ');
		label_size  = max(label_size, atoi(uStr::split(each_term[0],':')[1].c_str())	);
	}
	label_size += 1;
}


void DataSet::load_X(string target_path)
{
	X =	vector<VectorXd>();

	ifstream ifs( (target_path).c_str());
	int feature_index;
	
	VectorXd tmp;
	for(string line ; getline(ifs, line);)
	{
		tmp = VectorXd::Zero(feature_size);
		vector<string> each_term = uStr::split(line,' ');
		for(int i=0;i<each_term.size();i++ )
		{
			vector<string> fact = uStr::split(each_term[i] , ':');
			feature_index = atoi(fact[0].c_str());
			if(feature_index < feature_size)
				tmp(feature_index) = atof(fact[1].c_str());
		}
		tmp(feature_size-1) = 1;
		X.push_back(tmp);
	}
	ifs.close();
}

void DataSet::load_Y(string target_path)
{
	Y =	vector<VectorXd>();

	ifstream ifs( (target_path).c_str());
	
	VectorXd tmp;
	for(string line ; getline(ifs, line);)
	{
		tmp = VectorXd::Zero(label_size);
		vector<string> each_term = uStr::split(line,' ');
		vector<string> fact = uStr::split(each_term[0] , ':');
		tmp(atoi(fact[1].c_str())) = 1;
		Y.push_back(tmp);
	}
	ifs.close();
}


DataSet::DataSet(string target_dir)
{
	if(target_dir[target_dir.size()] != '/')
		target_dir += "/";
	init_Feature_Size(target_dir + "X");
	init_Label_Size(target_dir + "L");
}

void DataSet::set(string target_dir)
{
	if(target_dir[target_dir.size()] != '/')
		target_dir += "/";
	load_X(target_dir + "X");
	load_Y(target_dir + "L");
}

void DataSet::normalize()
{
	#pragma omp parallel for
	for(int i=0;i<X.size();i++)
		X[i] /= X[i].sum();		
}
