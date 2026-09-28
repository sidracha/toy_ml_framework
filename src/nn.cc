#include "nn.h"
#include "tensor.h"
#include "calc.h"

#include <cmath>
// ReLU is an operation and should return a Tensor, we also have to calculate the backward pass..
// for now its automatic
Tensor ReLU(const Tensor& t) {
	
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

Tensor Sigmoid (const Tensor& t) {
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


// ugh we have to do this odometer shit where N goes across the last row,
// and we apply it independently across each row....
// kind of annoying but whatveer man

Tensor softmax(const Tensor& t) {
	
	std::vector<int> shape = t.shape();
	std::vector<int> stride = t.stride();

	// how many odometers? well the last index is the one we want to iterate over... for 
	// the soft max. so
	
	int N_index = t.dim()-1;
	int N = shape[N_index];
	std::vector<int> odometer(N_index, 0);
	int odometer_index = 0;
	
	std::vector<double> output(t.tensor_node->data.size());
	
	while (odometer_index >= 0) {
		
		// i guess we get the index offset here,
		// thennnn we do the actual softmax here... by iterating i through 
		// all the row elements and write into the output at the end
		double e_sum = 0.0;
		for (int i=0; i<N; i++) {
			int data_index = odometer_index + i * stride[N_index];
			e_sum += std::exp(t.tensor_node->data[data_index]);	
		}

		// ok so we got the toal one, now we should calculate the per-
		// element softmax

		for (int i=0; i<N; i++) {
			int data_index = odometer_index + i * stride[N_index];
			output[data_index] = std::exp(t.tensor_node->data[data_index]) / e_sum;
		}
	
		odometer_index = odometer_next(odometer, shape, stride);

	}

	// ok so then we create the output node here...
	std::shared_ptr<TensorNode> node = make_operator_output_node(
		output, shape, {t.tensor_node}, softmax_backward);
	
	return Tensor(node);
	
}

// ok i derived it on paper its kind of annoying but whatveer
// let g be the upstream gradient row-wise vector

// and let z (i...n) be the input vector (row) into the row-wise
// softmax....
// we have L = f(y)
// y (i...n) = softmax(z)
// using the multivariate chain rule and some equation massaging on paper
//
// dL/dz_i = y_i * (g_i - sum(g_k * y_k) k0..n) 
// ok so yeah its O(n) thats basically the asnwer... first create
// the summation of g*y into a vector
// y is just the row-wise output so node->data i guess

void softmax_backward(TensorNode* node) {

	// how many odometers? well the last index is the one we want to iterate over... for 
	// the soft max. so
	
	// we only have one predecessor and its like the same
	// shape and such
	
	// so this is basically the "input z"
	TensorNode* A = node->predecessors[0].get();
	
	int N_index = node->dim()-1;
	int N = node->shape[N_index];
	std::vector<int> odometer(N_index, 0);
	int odometer_index = 0;
	
	while (odometer_index >= 0) {
		
		// just do everything per-row here
		// ok so first create the summation vector
		// with the uptream gradients
		
		double upstream_sum = 0.0;
		for (int i=0; i<N; i++) {
			int data_index = odometer_index + i * node->stride[N_index];
			upstream_sum += (node->grad[data_index] * node->data[data_index]);
		}

		for (int i=0; i<N; i++) {
			int data_index = odometer_index + i * node->stride[N_index];
			// write into the downstream z, which is A
			A->grad[data_index] += node->data[data_index] * (node->grad[data_index] - upstream_sum); 
		}

		odometer_index = odometer_next(odometer, node->shape, node->stride);

	}

}
