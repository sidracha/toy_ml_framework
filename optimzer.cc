#include "tensor.h"


// we do SIMPLE gradient descent
// nothing fanciful

// stores a vector list of layers...
// this will just hold the parameters
// can change this later to have a model class
class Optimzer {
public:
	double lr;
	vector<Layer*> layers;

	Optimizer(double _lr) : lr(_lr) {}
	
	zero_grad() {
		// go through every single tensor in layers.. and set it to 0
		// but also each layer should have the interface to set to 0
	}

}
