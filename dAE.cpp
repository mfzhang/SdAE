#include "dAE.h"

dAE::dAE(Param &p)
{
	param = &p;
}

double dAE::loss(VectorXd &x)
{
	double result = 0;
	
	VectorXd y = x_to_y(x);
	VectorXd z_margin = W.transpose()*y + b_observe;
	
	#pragma omp parallel for reduction(+:result)
	for(int j=0;j<observe_size;j++)
		result 	-= x(j) 		* uMath::Ln_Sigmoid(  z_margin(j) )
				 + (1 - x(j))	* uMath::Ln_Sigmoid( -z_margin(j) );
	return result;
}


double dAE::Loss(vector<VectorXd> &X)
{
	double result = 0;
	#pragma omp parallel for reduction(+:result)
	for(int i=0;i<X.size();i++)
		result += loss(X[i]);
	return result;
}

VectorXd dAE::x_to_y(VectorXd &x)
{
	VectorXd result = VectorXd::Zero(hidden_size);
	#pragma omp parallel for
	for(int i=0;i<result.size();i++)
		result(i) = uMath::Sigmoid(W.row(i).dot(x) + b_hidden(i) ) ;
	return result;
}

VectorXd dAE::y_to_z(VectorXd &y)
{
	VectorXd result = VectorXd::Zero(observe_size);
	#pragma omp parallel for
	for(int i=0;i<result.size();i++)
		result(i) = uMath::Sigmoid( W.col(i).dot(y) + b_observe(i));
	return result;
}

VectorXd dAE::corrupt(VectorXd &x)
{
	VectorXd result = x;
	#pragma omp parallel for
	for(int i=0; i<x.size(); i++)
		if( result(i) != 0 && rand() / (RAND_MAX + 1.0) > param->Dropout_rate)
			result(i) = 0;
	return result;
}

void dAE::update(VectorXd &x,double &learning_rate)
{
	vector<VectorXd> corrupted_xs = vector<VectorXd>(param->Dropout_time);
	#pragma omp parallel for
	for(int i=0;i<param->Dropout_time;i++)
		corrupted_xs[i] = corrupt(x);

	VectorXd y	= VectorXd::Zero(hidden_size);
	VectorXd z	= VectorXd::Zero(observe_size);
	VectorXd tmp	= VectorXd::Zero(hidden_size);

	for(int k=0;k<param->Dropout_time;k++)
	{
//*
		#pragma omp parallel for
		for(int i=0;i<y.size();i++)
			y(i) = uMath::Sigmoid(W.row(i).dot(corrupted_xs[k]) + b_hidden(i));
		
		#pragma omp parallel for
		for(int i=0;i<z.size();i++)
			z(i) = learning_rate * (x(i) - uMath::Sigmoid( W.col(i).dot(y) + b_observe(i)));

		#pragma omp parallel for
		for(int i=0;i<y.size();i++)
			tmp(i) = W.row(i).dot(z) * (1-y(i)) * y(i);
		
		#pragma omp parallel for
		for(int i=0;i<b_observe.size();i++)
			b_observe(i)	+= z(i);

		#pragma omp parallel for
		for(int i=0;i<b_hidden.size();i++)
			b_hidden(i) += tmp(i);

		#pragma omp parallel for
		for(int j=0;j<y.size();j++)
			#pragma omp parallel for
			for(int i=0;i<z.size();i++)
				W(j,i) += tmp(j) * corrupted_xs[k](i) + y(j) * z(i);
//*/
/*
		y = x_to_y(corrupted_xs[k]);
		z = y_to_z(y);
		
		tmp = W*(x-z);
		tmp = tmp.array() * y.array() * (1 - y.array());

		b_observe	+= learning_rate * (x-z);
		b_hidden += learning_rate * tmp;
		W += learning_rate (tmp * corrupted_xs[k].transpose() + y * z.transpose());
//*/
	}
}

void dAE::train(vector<VectorXd> &X,int _hidden_size)
{
	observe_size 	= X[0].size();
	hidden_size	= _hidden_size;

	W		= MatrixXd::Zero(hidden_size,observe_size);
	b_hidden 	= VectorXd::Zero(hidden_size);
	b_observe = VectorXd::Zero(observe_size);

	vector<int> random_index;
	for(int i=0;i<X.size();i++)
		random_index.push_back(i);
	
	double	previous_Loss	= INT_MAX;
	double	current_Loss	= INT_MAX-1;
	double	diff_Loss;
	double	learning_rate;
	
	for(int count=1;count<param->dAE_max_iteration+1;count++)
	{
		random_shuffle(random_index.begin(),random_index.end());
		learning_rate = 1.0/(count*X.size());
//		learning_rate = 1.0/(count);
		for(int j=0;j< X.size();j++)
			update(X[random_index[j]], learning_rate);
		
		previous_Loss	= current_Loss;
		current_Loss	= Loss(X);
		diff_Loss		= previous_Loss - current_Loss;
		cout << "\t"	<< count
			<< "\t"	<< uStr::to_String_Padding(diff_Loss,14)
			<< "\t"	<< uStr::to_String_Padding(current_Loss,14)
			<< endl;
		if(abs(diff_Loss) < param->dAE_stop_threshold )
			break;
	}
}

VectorXd dAE::get_hidden_value(VectorXd &x)
{
	return x_to_y(x);
}

VectorXd dAE::get_propagated_error(VectorXd &y)
{
	return W.transpose() * y;
}

void dAE::back_propagate(VectorXd &input_value,VectorXd &error_value,double &learning_rate)
{
	W		+= learning_rate * error_value * input_value.transpose();
	b_hidden	+= learning_rate * error_value;
}



////////////////////////////////////////////////////////////////////////////////


void dAE::dump_param()
{
	cout << "dump_param" << endl;
	
	cout << "\\tdump b_hidden" << endl;
	uIO::dump("b_hidden",	b_hidden);
	
	cout << "\tdump b_observe" << endl;
	uIO::dump("b_observe",	b_observe);
	
	cout << "\tdump W" << endl;
	uIO::dump("W",W);
	
	cout << "done" << endl;
}

void dAE::dump_reconstruct(vector<VectorXd> &X)
{
	cout << "dump_param" << endl;

	cout << "\tdump X_observed" << endl;
	uIO::dump("X_observed",	X);

	cout << "\treconstrunt X" << endl;
	vector<VectorXd> reconstructed = vector<VectorXd>(X.size());
	#pragma omp prallel for
	for(int i=0;i<X.size();i++)
	{
		VectorXd y = x_to_y(X[i]);
		reconstructed[i] = y_to_z(y);
	}
	
	cout << "\tdump X_reconstruct" << endl;
	uIO::dump("X_reconstructed", reconstructed);

	cout << "done" << endl;
}

