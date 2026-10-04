#include "tensor.h"
#include "layer.h"

#include <vector>

class Optimizer {
public:
	double lr;
	std::vector<Tensor*> params;
	
	Optimizer(std::vector<Tensor*> _params, double _lr) : params(_params), lr(_lr) {}
	//Optimizer(std::vector<std::unique_ptr<Layer>>& _layers, double _lr) : lr(_lr), layers(_layers) {}
	void zero_grad();
	virtual void step() = 0;
};

class SGDOptimizer : public Optimizer {
public:
	SGDOptimizer(std::vector<Tensor*> _params, double _lr) : Optimizer(_params, _lr) {}
	void step() override;
};

class Adam : public Optimizer {
public:
	// for each param (which is a tensor) store
	// a vector for m/w corresponding to the weight
	
	double beta1;
	double beta2;
	double eps;

	// first and second moments
	std::vector<std::vector<double>> m;
	std::vector<std::vector<double>> v;
	
	// initialize the betas/eps with the default ones from the paper
	Adam(std::vector<Tensor*> _params, double _lr, double _beta1=0.9, double _beta2=0.999, double _eps=1e-8) :
		Optimizer(_params, _lr),
		beta1(_beta1),
		beta2(_beta2),
		eps(_eps) {
		
			// ok so here we want to actually set up the m and v
			// we set them to 0 first as initalization
			// BUT we want to set the size of them depending on their 
			// param sizes

			for (Tensor* t : params) {
				// get each individual tensor
				// create the moments for each weight inside of that tensor
				m.push_back(std::vector<double>(t->tensor_node->grad.size(), 0));
				v.push_back(std::vector<double>(t->tensor_node->grad.size(), 0));

			}

	}

	void step() override;

};
