#ifndef __UTILITY_
#define __UTILITY_

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>
#include <sys/stat.h>

#include <Eigen/Eigen>

using namespace Eigen;
using namespace std;

namespace uStr
{
	string to_String_Padding(double ,int );
	string to_String_Padding(int ,int );
	vector<string> split(const string &, char );
}

namespace uIO
{
	void make_directory(string );
	void dump(string ,VectorXd &);
	void dump(string ,MatrixXd &);
	void dump(string ,vector<VectorXd> &);
}

namespace uMath
{
	double Sigmoid(double );
	double Sqrt_Sigmoid_Derivation(double );
	double Ln_Sigmoid(double );
	VectorXd	softmax(VectorXd &);
}


#endif
