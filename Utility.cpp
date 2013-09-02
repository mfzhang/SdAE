#include "Utility.h"

namespace uStr
{
	string to_String_Padding(int a,int b)
	{
		b++;
		ostringstream ss;
		ss << a;
		string result = ss.str();
		if(a >= 0)
			result =  " " + result;
		if(result.length() <= b)
			while(result.length() != b)
				result += " ";
		else
			result = result.substr(0,b);
		return result;
	}

	string to_String_Padding(double a,int b)
	{
		b++;
		ostringstream ss;
		ss << a;
		string result = ss.str();
		if(a >= 0)
			result =  " " + result;
		if(result.length() <= b)
			while(result.length() != b)
				result += " ";
		else
			result = result.substr(0,b);
		return result;
	}

	vector<string> split(const string &s, char delim)
	{
		vector<string> elems;
		stringstream ss(s);
		for(string item ; getline(ss, item, delim) ; )
			elems.push_back(item);
		return elems;
	}
}

namespace uIO
{
	void make_directory(string dir_name)
	{
		struct stat st;
		int ret = stat(dir_name.c_str(), &st);
		if (0 != ret)
			mkdir(dir_name.c_str(),S_IRWXU | S_IRWXG | S_IRWXO);
	}
	void dump(string file_name,VectorXd &v)
	{
		make_directory("DUMP");
		ofstream out(("DUMP/" + file_name).c_str(),ios::app);
		out << v.transpose() << endl;
		out.close();
	}
	void dump(string file_name,MatrixXd &x)
	{
		make_directory("DUMP");
		ofstream out(("DUMP/" + file_name).c_str(),ios::app);
		out << x.transpose() << endl;
		out.close();
	}
	void dump(string file_name,vector<VectorXd> &X)
	{
		make_directory("DUMP");
		ofstream out(("DUMP/" + file_name).c_str(),ios::app);
		ostringstream ss;
		for(int i=0;i<X.size();i++)
			ss << X[i].transpose() << endl;
		out << ss;
		out.close();
	}
}

namespace uMath
{
	double Sigmoid(double x)
	{
		return (tanh(x/2)+1)/2;
	}

	double Sqrt_Sigmoid_Derivation(double x)
	{
		return 1/(2*cosh(x/2));
	}

	double Ln_Sigmoid(double x)
	{
		if(x < -32)	return x;
		else			return -log( 1+exp(-x) );
	}

	VectorXd softmax(VectorXd &x)
	{
		VectorXd result = x;
		double max = x.maxCoeff();
		#pragma omp parallel for
		for(int i=0; i<result.size(); i++)
			result[i] = exp(result[i] - max);
		result/=result.sum();
		return result;
	}
}


