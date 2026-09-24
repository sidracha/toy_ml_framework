#include "linear.h"
#include "nn.h"

#include <vector>
#include <stdexcept>

Tensor Linear::forward(Tensor t) {
	t = t.MATMUL_2D_ADD(weight);
	t = t.BIAS_ADD_2D_1D(bias);
	return t;
}

void Linear::zero_grad() {
	
	// iterate through the weights and the biases
	
	// set weights grad to 0
	for (int i=0; i<weight.tensor_node->grad.size(); i++) {
		weight.tensor_node->grad[i] = 0.0;
	}

	// set biases grad to 0
	for (int i=0; i<bias.tensor_node->grad.size(); i++) {
		bias.tensor_node->grad[i] = 0.0;
	}

}

// apply negative gradient scaled by eta
void Linear::gradient_descent_step(double lr) {
	
	// iterate through the weights
	for (int i=0; i<weight.tensor_node->grad.size(); i++) {
		double grad_step = -lr * weight.tensor_node->grad[i];
		weight.tensor_node->data[i] += grad_step;
	}
	
	// the biases
	for (int i=0; i<bias.tensor_node->grad.size(); i++) {
		double grad_step = -lr * bias.tensor_node->grad[i];
		bias.tensor_node->data[i] += grad_step;
	}

}

void Linear::clear_graph() {
	weight.tensor_node->predecessors = {};
	bias.tensor_node->predecessors = {};

}

Tensor LinearReLU::forward(Tensor t) {
	t = t.MATMUL_2D_ADD(weight);
	t = t.BIAS_ADD_2D_1D(bias);
	t = ReLU(t);
	return t;
}
