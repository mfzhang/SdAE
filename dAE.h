#ifndef _DENOISING_AUTOENCODER_
#define _DENOISING_AUTOENCODER_


#include <vector>
#include <math.h>
#include <limits.h>
#include <omp.h>

#include "Param.h"
#include "Utility.h"

using namespace Eigen;
using namespace std;

class dAE
{
private:
	Param 	*param;
	int		observe_size;
	int 		hidden_size;
	
	MatrixXd	W;
	VectorXd 	b_hidden;
	VectorXd 	b_observe;

	double	loss(VectorXd &);
	double 	Loss(vector<VectorXd> &);
	
	VectorXd 	corrupt(VectorXd &);
	VectorXd 	x_to_y(VectorXd &);
	VectorXd 	y_to_z(VectorXd &);
	void		update(VectorXd &,double &);
	
public:
			dAE(){};
			dAE(Param &);
	void 	train(vector<VectorXd> &,int );
	
	VectorXd	get_hidden_value(VectorXd &);
	VectorXd	get_propagated_error(VectorXd &);
	void		back_propagate(VectorXd &,VectorXd &,double &);
	
	void		dump_param();
	void		dump_reconstruct(vector<VectorXd> &);
	void		test_L1_Loss(vector<VectorXd> &);
	void		test_L2_Loss(vector<VectorXd> &);
};

#endif
