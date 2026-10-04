#include "tensor.h"
#include "layer.h"
#include "optimizer.h"

#include <cmath>

// stores a vector list of params
// this will just hold the parameters

	
void Optimizer::zero_grad() {
	// okkk lets just go through all of the tenosr params 
	// and do it
	for (Tensor* tensor : params) {
		int N = tensor->tensor_node->grad.size();
		for (int i=0; i<N; i++) tensor->tensor_node->grad[i] = 0;
	}
}

// for the sgd one just iterate through
// and apply the parameter updaate thats it
void SGDOptimizer::step() {
	
	for (Tensor* tensor : params) {
		int N = tensor->tensor_node->grad.size();
		for (int i=0; i<N; i++) {
			tensor->tensor_node->data[i] -= lr * tensor->tensor_node->grad[i];
		}
	}
}

// apply the algorithm from the 
// Adam 2015 iclr paper
// we already have m and v from the init 
void Adam::step() {
		
	t++;

	double beta1_scale = 1-beta1;
	double beta2_scale = 1-beta2;

	double beta1_powscale = 1 - std::pow(beta1, (double) t);
	double beta2_powscale = 1 - std::pow(beta2, (double) t);
	// iterate over each tensor
	for (int i=0; i<params.size(); i++) {
		
		Tensor* tensor = params[i];
		int N = tensor->tensor_node->grad.size();
		for (int j=0; j<N; j++) {
			// apply the algo here, each pointwise so we are good
			double g = tensor->tensor_node->grad[j];

			double mt = beta1*m[i][j] + (beta1_scale) * g;
			double vt = beta2*v[i][j] + (beta2_scale) * (g*g);
			// bias corrected estimate
			double mc = mt / (beta1_powscale);
			double vc = vt / (beta2_powscale);

			// update parameter
			// find the update amount
			double step_amount = mc / (std::sqrt(vc) + eps);
			double update = -lr * step_amount;
			tensor->tensor_node->data[j] += update;

			// now store the values back
			m[i][j] = mt;
			v[i][j] = vt;

		}

	}

}
