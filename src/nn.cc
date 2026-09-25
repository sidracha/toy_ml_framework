#include "nn.h"
#include "tensor.h"

#include <cmath>
// ReLU is an operation and should return a Tensor, we also have to calculate the backward pass..
// for now its automatic
Tensor ReLU(Tensor t) {
	
	// we have to make sure the shape and the stride are the same... but when we create it with the shape,
	// it might be discontinuous...
	// do we create the gradient...?
	
	// i guess we can set the stride the same i guess...
	// create a tensor with the same shape and stride...
	
	Tensor new_tensor = create_tensor_clone_scalar(t, 0.0);
	// now its a clone, do the ReLU
	for (int i=0; i<t.tensor_node->data.size(); i++) {
		int val;
		if (t.tensor_node->data[i] > 0.0) new_tensor.tensor_node->data[i] = 1.0; 
	}

	return t * new_tensor;

}

double sigmoid(double x) {
	return 1.0 / (1.0 + std::exp(-x));
}

Tensor Sigmoid (Tensor t) {
	// I guess this is just a pointwise operation
	// we have to create a new node for this... but will it be within the class?
	// we need some extendable way to do this
	std::vector<double> data(t.tensor_node->data.size());
	for (int i=0; i<t.tensor_node->data.size(); i++) {
		// do the sigmoid operation
		double x = t.tensor_node->data[i];
		data[i] = sigmoid(x);
	}
	// create the output node
	std::shared_ptr<TensorNode> node = make_operator_output_node(
										data, 
										t.shape(), 
										{t.tensor_node}, 
										Sigmoid_backward);
	
	return Tensor(node);
	
}


void Sigmoid_backward(TensorNode* node) {
	
	// this is a unary operator, so i guess we only have one predecessor...
	// but what? how do we register this as an operation...
	// just call sigmoid squared... that is the gradient... 
	// get the first parent
	TensorNode* a = node->predecessors[0].get();
	for (int i=0; i<a->data.size(); i++) {
		a->grad[i] += (node->data[i] * (1.0-node->data[i]) * node->grad[i]);
	}

}
