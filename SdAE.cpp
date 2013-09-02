#include "SdAE.h"

SdAE::SdAE(Param &p)
{
	param = &p;
}

void SdAE::pretrain(vector<VectorXd> X,vector<VectorXd> &Y)
{
	cout << "Pretraining" << endl;
	
	daes = new dAE[param->Layer_Size];
	for(int i=0;i<param->Layer_Size;i++)
	{
		cout << "  Layer " << i << endl;
		
		daes[i] = dAE(*param);
		daes[i].train(X, param->Node_Sizes[i]);
		
		#pragma omp parallel for
		for(int j=0;j<X.size();j++)
			X[j] = daes[i].get_hidden(X[j]);
	}
	cout << "  Last Layer" << endl;
	lr = new LR(*param);
	lr->train(X,Y);
}

void SdAE::back_propagate(VectorXd &x,VectorXd &y,double &learning_rate)
{
	vector<VectorXd> layer_values = vector<VectorXd>(param->Layer_Size);
	layer_values[0] = daes[0].get_hidden(x);
	for(int i=1;i<param->Layer_Size;i++)
		layer_values[i] = daes[i].get_hidden(layer_values[i-1]);
	
	VectorXd error_value	= lr->get_label_margin(layer_values[param->Layer_Size-1]) - y;
	VectorXd tmp			= lr->get_input_margin(error_value);
	lr->back_propagate(layer_values[param->Layer_Size-1],error_value,learning_rate);
	
	for(int i=param->Layer_Size-1;0<i;i--)
	{
		error_value = layer_values[i].array()*(1-layer_values[i].array()) * tmp.array();
		daes[i].back_propagate(layer_values[i],error_value ,learning_rate);
		tmp = daes[i].get_observe_margin(error_value);
	}
	error_value = layer_values[0].array()*(1-layer_values[0].array()) * tmp.array();
	daes[0].back_propagate(x,error_value ,learning_rate);
}

double SdAE::loss(VectorXd &x,VectorXd &y)
{
	VectorXd layer_value = daes[0].get_hidden(x);
	for(int j=1;j<param->Layer_Size;j++)
		layer_value = daes[j].get_hidden(layer_value);
	return lr->get_loss(layer_value,y);
}

double SdAE::Loss(vector<VectorXd> &X,vector<VectorXd> &Y)
{
	double result = 0;
	#pragma omp parallel for reduction(+:result)
	for(int i=0;i<X.size();i++)
		result += loss(X[i],Y[i]);
	return result;
}

void SdAE::finetune(vector<VectorXd> &X,vector<VectorXd> &Y)
{
	if(param->Layer_Size == 0)
		return;
	cout << "Finetuning" << endl;

	vector<int> random_index;
	for(int i=0;i<X.size();i++)
		random_index.push_back(i);
	
	double	previous_Loss	= INT_MAX;
	double	current_Loss	= INT_MAX-1;
	double	diff_Loss;
	double	learning_rate;
	
	for(int count=1;count<param->BP_max_iteration+1;count++)
	{
		random_shuffle(random_index.begin(),random_index.end());
		learning_rate = 1.0/(count*X.size());
		for(int j=0;j<X.size();j++)
			back_propagate(X[random_index[j]],Y[random_index[j]],learning_rate);
		
		previous_Loss = current_Loss;
		current_Loss  = Loss(X,Y);
		diff_Loss		= previous_Loss - current_Loss;
		cout << "\t"	<< count
			<< "\t"	<< uStr::to_String_Padding(diff_Loss,14)
			<< "\t"	<< uStr::to_String_Padding(current_Loss,14)
			<< endl;
		if(abs(diff_Loss) < param->BP_stop_threshold )
			break;
	}
}


/////////////////////////////////////////////////////////////////////////////////////


void SdAE::test(vector<VectorXd> &X,vector<VectorXd> &Y)
{
	for(int i=0;i<param->Layer_Size;i++)
		#pragma omp parallel for
		for(int j=0;j<X.size();j++)
			X[j] = daes[i].get_hidden(X[j]);
	#pragma omp parallel for
	for(int i=0;i<X.size();i++)
		X[i] = lr->get_label_margin(X[i]);
	
	double result = 0;
	#pragma omp parallel for reduction(+:result)
	for(int i=0;i<X.size();i++)
	{
		double	max_value = X[i].maxCoeff();
		for(int j=0;j<X[i].size();j++)
			if(max_value == X[i](j) && Y[i](j) != 1)
			{
				result++;
				break;
			}
	}
	
	cout	<< "Error\t"		<< result/X.size()
		<< "=" << result	<< "/" << X.size()
		<< endl;
}
