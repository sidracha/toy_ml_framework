#include "tensor.h"
#include "losses.h"

#include <cmath>
#include <stdexcept>
#include <iostream>

// make the stride and stuff the same
Tensor MSELoss(Tensor a, Tensor b) {
	
	if (a.shape() != b.shape() || a.stride() != b.stride()) {
		throw std::runtime_error("Tensor shapes do not match");
	}
	// each row holds a "vector"
	// so I guess calculate the MSE of the vectors first
	
	// it should create a 1-D tensor at the end.. and we need to attach a backwards fucntion
	// how can we do this?
	
	// this is JUST the forward pass I guess
	std::vector<double> output(1);
	// we can do it in blocks ok but this will be left as a TODO
	// for now just sum all of it...
	// worst case the output tensor wont be that bad right
	// like in bubbleML its like 32 * 5 * 4 * 512 * 512 with a value of 1
	// in practice its less than that but thats like less than an int
	// so should be fine
	
	double sum = 0;
	for (int i=0; i<a.tensor_node->data.size(); i++) {
		double diff = (a.tensor_node->data[i] - b.tensor_node->data[i]);
		sum += diff*diff;
	}
	sum /= a.tensor_node->data.size();
	output[0] = sum;

	//we have to make the node, I guess?
	std::shared_ptr<TensorNode> node = make_operator_output_node(
			output, 
			{1}, 
			{a.tensor_node, b.tensor_node}, 
			MSELoss_backward);
	
	return Tensor(node);
}


// how to calcualte the backward pass...???
// this is pretty easy, actually.
// lets fuse the nodes too, I guess

// dda = (2/n) * (a-b)
// ddb = -(2/n) * (a-b)
void MSELoss_backward(TensorNode* node) {

	TensorNode* a = node->predecessors[0].get();
	TensorNode* b = node->predecessors[1].get();
	std::cout << "MSE backward - a pointer: " << a << " a->grad.size(): " << a->grad.size() << std::endl;
	int N = a->data.size();
	double first_term, second_term, dda, ddb;
	for (int i=0; i<N; i++) {
		first_term = 2 / static_cast<double>(N);
		second_term = (a->data[i] - b->data[i]);
		dda = node->grad[0] * first_term * second_term;
		ddb = -1.0 * node->grad[0] * first_term * second_term;
		a->grad[i] += dda;
		b->grad[i] += ddb;
	}
	std::cout << "MSE backward - a->grad[0] after write: " << a->grad[0] << std::endl;
}
