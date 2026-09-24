#include "tensor.h"
#include "layer.h"

#include <vector>

class Optimizer {
public:
	double lr;
	std::vector<Layer>* layers;

	Optimizer(double _lr) : lr(_lr) {}
	void zero_grad();
	void step();
}

