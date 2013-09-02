#include "LR.h"

double LR::loss(VectorXd &x,VectorXd &y)
{
	double result = 0;
	VectorXd margin = W.transpose() * x;
	margin = margin.array() - margin.maxCoeff();
	for(int i=0;i<margin.size();i++)
		if(y(i) == 1)
			result -= margin(i);
	margin = margin.array().exp();
	return result + log( margin.sum() );
}


double LR::Loss(vector<VectorXd> &X,vector<VectorXd> &Y)
{
	double result = 0;
	#pragma omp parallel for reduction(+:result)
	for(int i=0;i<X.size();i++)
		result += loss(X[i],Y[i]);
	return result;
}

void LR::update(VectorXd &x,VectorXd &y,double learning_rate)
{
	VectorXd margin = W.transpose() * x + b;
	margin = uMath::softmax(margin) - y;
	b -= learning_rate * margin;
	
	#pragma omp parallel for
	for(int i=0;i<margin.size();i++)
		W.col(i) -= learning_rate*( margin[i]*x - param->Alpha * W.col(i) );
}

LR::LR(Param &p)
{
	param = &p;
}

void LR::train(vector<VectorXd> &X,vector<VectorXd> &Y)
{
	W	= MatrixXd::Zero(X[0].size(),Y[0].size());
	b	= VectorXd::Zero(Y[0].size());
	
	vector<int> random_index;
	for(int i=0;i<X.size();i++)
		random_index.push_back(i);

	double	previous_Loss	= INT_MAX;
	double	current_Loss	= INT_MAX-1;
	double	diff_Loss;
	double	learning_rate;
	
	for(int count=1;count < param->LR_max_iteration+1;count++)
	{
		random_shuffle(random_index.begin(),random_index.end());
		learning_rate = 1.0/(count*X.size());
		for(int i=0;i<X.size();i++)
			update(X[random_index[i]],Y[random_index[i]],learning_rate);
		
		previous_Loss	= current_Loss;
		current_Loss	= Loss(X,Y);
		diff_Loss		= previous_Loss - current_Loss;
		cout << "\t"	<< count
			<< "\t"	<< uStr::to_String_Padding(diff_Loss,14)
			<< "\t"	<< uStr::to_String_Padding(current_Loss,14)
			<< endl;
		if(abs(diff_Loss) < param->LR_stop_threshold )
			break;
	}
}

double LR::get_loss(VectorXd &x,VectorXd &y)
{
	return loss(x,y);
}

VectorXd LR::get_label_margin(VectorXd &x)
{
	return W.transpose() * x;
}

VectorXd LR::get_input_margin(VectorXd &y)
{
	return W * y;
}

void LR::back_propagate(VectorXd &input_value,VectorXd &error_value,double &learning_rate)
{
	W	-= learning_rate * input_value * error_value.transpose();
	b	-= learning_rate * error_value;
}

