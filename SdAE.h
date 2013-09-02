#ifndef _STACKED_DAE_
#define _STACKED_DAE_

#include <omp.h>
#include <limits.h>

#include <Eigen/Eigen>

#include "Utility.h"
#include "Param.h"
#include "LR.h"
#include "dAE.h"

using namespace std;
using namespace Eigen;

class SdAE
{
private:
	Param	*param;
	dAE		*daes;
	LR		*lr;
	
	void		back_propagate(VectorXd &,VectorXd &,double &);
	double	loss(VectorXd &,VectorXd &);
	double	Loss(vector<VectorXd> &,vector<VectorXd> &);
	
public:
			SdAE(Param &);
	void		pretrain(vector<VectorXd> ,vector<VectorXd> &);
	void		finetune(vector<VectorXd> &,vector<VectorXd> &);
	void		test(vector<VectorXd> &,vector<VectorXd> &);
};

#endif
