#include "Param.h"

Param::Param(){}

void Param::dump()
{
	cout << "Param Dump"	<< endl;
	
	cout << "\tAlpha\t\t"		<< Alpha				<< endl
		
		<< "\tDropout_time\t"	<< Dropout_time		<< endl
		<< "\tDropout_rate\t"	<< Dropout_rate		<< endl
		
		<< "\tdAE_max_iteration"	<< dAE_max_iteration	<< endl
		<< "\tBP_max_iteration"	<< BP_max_iteration		<< endl
		<< "\tLR_max_iteration"	<< LR_max_iteration		<< endl

		<< "\tdAE_stop_threshold"<< dAE_stop_threshold	<< endl
		<< "\tBP_stop_threshold"	<< BP_stop_threshold	<< endl
		<< "\tLR_stop_threshold"	<< LR_stop_threshold	<< endl
		
		<< "\tLayer_Sizes\t"	<< Layer_Size			<< endl;
	for(int i=0;i<Layer_Size;i++)
		cout << "\t\tNode " << i  << "\t" << Node_Sizes[i] << endl;
}

Param::Param(int argc,char* argv[])
{
	Alpha 		= 1e-5;
	
	Dropout_time 	= 1;
	Dropout_rate	= 0.8;
	
	dAE_max_iteration 	= 10;
	BP_max_iteration	= 100;
	LR_max_iteration	= 100;
	
	dAE_stop_threshold	= 1e-0;
	BP_stop_threshold	= 1e-6;
	LR_stop_threshold	= 1e-3;
	
	Node_Sizes.push_back(10);
	Layer_Size = Node_Sizes.size();
	
	for(int i=1;i != argc;)
	{
		if(!strcmp(argv[i],"-A"))
		{
			Alpha = atof(argv[i+1]);
			i+=2;
			continue;
		}
		
		//Dropout param
		else	if(!strcmp(argv[i],"-DR"))
		{
			Dropout_rate = atof(argv[i+1]);
			i+=2;
			continue;
		}
		else if(!strcmp(argv[i],"-DT"))
		{
			Dropout_time = atof(argv[i+1]);
			i+=2;
			continue;
		}
		
		//Iterate param
		else if(!strcmp(argv[i],"-DAI"))
		{
			dAE_max_iteration = atof(argv[i+1]);
			i+=2;
			continue;
		}
		else if(!strcmp(argv[i],"-BPI"))
		{
			BP_max_iteration = atof(argv[i+1]);
			i+=2;
			continue;
		}
		else if(!strcmp(argv[i],"-LRI"))
		{
			LR_max_iteration = atof(argv[i+1]);
			i+=2;
			continue;
		}

		//Threshold param
		else if(!strcmp(argv[i],"-DAT"))
		{
			dAE_stop_threshold = atof(argv[i+1]);
			i+=2;
			continue;
		}
		else if(!strcmp(argv[i],"-BPT"))
		{
			BP_stop_threshold = atof(argv[i+1]);
			i+=2;
			continue;
		}
		else if(!strcmp(argv[i],"-LRT"))
		{
			LR_stop_threshold = atof(argv[i+1]);
			i+=2;
			continue;
		}
		
		//Node Sizes param
		else if(!strcmp(argv[i],"-NS"))
		{
			Node_Sizes = vector<int>();
			i++;
			while(i != argc && argv[i][0] != '-')
			{
				Node_Sizes.push_back(atoi(argv[i]));
				i++;
			}
			Layer_Size = Node_Sizes.size();
			continue;
		}
		else
		{
			cout << "Can't find Such Option\t" << argv[i] << endl;
			i++;
		}
	}
}
