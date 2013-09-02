#ifndef _LOGISTIC_REGRESSION_
#define _LOGISTIC_REGRESSION_

#include <vector>
#include <math.h>
#include <unistd.h>
#include <limits.h>
#include <omp.h>

#include "Param.h"
#include "Utility.h"

#include <Eigen/Eigen>

using namespace Eigen;
using namespace std;

class LR
{
private:
	Param 	*param;
	
	MatrixXd 	W;
	VectorXd	b;
	
	double	loss(VectorXd &,VectorXd &);
	double	Loss(vector<VectorXd> &,vector<VectorXd> &);
	void		update(VectorXd &,VectorXd &,double );
	
public:
	
			LR(Param &);
	void		train(vector<VectorXd> &,vector<VectorXd> &);

	double	get_loss(VectorXd &,VectorXd &);
	VectorXd	get_label_margin(VectorXd &);
	VectorXd	get_input_margin(VectorXd &);
	void		back_propagate(VectorXd &,VectorXd &,double &);
	
};

#endif
