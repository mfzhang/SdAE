#include <iostream>
#include <omp.h>
#include <time.h>

#include <Eigen/Eigen>

#include "Param.h"
#include "DataSet.h"
#include "SdAE.h"

using namespace std;

void init()
{
	srand(time(NULL));
	Eigen::initParallel();
	Eigen::setNbThreads(10);
	#ifdef _OPENMP
		omp_set_num_threads(10);
		omp_set_nested(10);
	#endif
}

void test_SdAE(Param &param)
{
	DataSet data("TRAIN");
	data.set("TRAIN");
	data.normalize();

	SdAE sdae(param);
	sdae.pretrain(data.X,data.Y);
	sdae.finetune(data.X,data.Y);
	
	data.set("TEST");
	data.normalize();

	sdae.test(data.X,data.Y);
}

void test_dAE(Param &param)
{
	if(param.Layer_Size == 0)
		return;
	
	DataSet data("TRAIN");
	data.set("TRAIN");
	data.normalize();

	dAE dae(param);
	dae.train(data.X,param.Node_Sizes[0]);
	
	dae.dump_param();
	dae.dump_reconstruct(data.X);
	
}

int main(int argc, char* argv[])
{
	init();
	Param param(argc, argv);
<<<<<<< HEAD
	test_dAE(param);
=======
	test_SdAE(param);
>>>>>>> origin/master
}


