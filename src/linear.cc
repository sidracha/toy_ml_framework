#include "linear.h"
#include "nn.h"

#include <vector>
#include <stdexcept>

Tensor Linear::forward(Tensor t) {
	t = t.MATMUL_2D(params[0]);
	t = t + params[1];
	return t;
}

Tensor LinearReLU::forward(Tensor t) {
	t = t.MATMUL_2D(params[0]);
	t = t + params[1];
	t = ReLU(t);
	return t;
}

Tensor LinearSigmoid::forward(Tensor t) {
	t = t.MATMUL_2D(params[0]);
	t = t + params[1];
	t = Sigmoid(t);
	return t;
}
