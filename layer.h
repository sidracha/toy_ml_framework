#pragma once

#include "tensor.h"

#include <vector>


// layer also has to implement set the grad to 0
class Layer {
public:
	Layer() {}

	virtual Tensor forward(Tensor t) = 0;
	virtual void zero_grad() = 0;
	virtual void gradient_descent_step(double lr) = 0;

};
