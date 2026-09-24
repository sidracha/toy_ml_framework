#include "tensor.h"
#include "layer.h"

#include <vector>

class Optimizer {
public:
	double lr;
	std::vector<std::unique_ptr<Layer>>& layers;

	Optimizer(std::vector<std::unique_ptr<Layer>>& _layers, double _lr) : lr(_lr), layers(_layers) {}
	void zero_grad();
	void step();
};

