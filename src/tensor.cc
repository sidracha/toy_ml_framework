#include "tensor.h"
#include "calc.h"
#include "tensor_grad.h"

#include <vector>
#include <deque>
#include <stdexcept>
#include <cmath>
#include <random>
#include <iostream>

class TensorShapeMismatch : public std::runtime_error {
public:
	TensorShapeMismatch(Tensor a, Tensor b) : std::runtime_error("Tensor shapes are mismatched") {};
};

class InvalidTensorShape : public std::runtime_error {
public:
	InvalidTensorShape() : std::runtime_error("Invalid Tensor Shape") {};
};

void TensorNode::increase_indegree() {
	indegree++;
}


void TensorNode::permute(const std::vector<int>& index_after) {
	std::vector<int> shape_after(dim());
	std::vector<int> stride_after(dim());
	for (int i=0; i<index_after.size(); i++) {
		shape_after[i] = shape[index_after[i]];
		stride_after[i] = stride[index_after[i]];
	}
	stride = stride_after;
	shape = shape_after;
}

// do a 2D transpose of the last 2 elements
void TensorNode::transpose() {
	int n = dim();
	if (n < 2) throw std::runtime_error("Tranpose called with dim < 2");
	std::vector<int> permute_vector(n);
	for (int i=0; i<n-2; i++) permute_vector[i] = i;
	// the last 2 elements after flipped
	permute_vector[n-1] = n-2;
	permute_vector[n-2] = n-1;
	permute(permute_vector);

}


std::shared_ptr<TensorNode> create_tensor_node(std::vector<double>& data, const std::vector<int>& shape) {
	
	std::shared_ptr<TensorNode> node = std::make_shared<TensorNode>(data, shape);
	return node;
}

std::shared_ptr<TensorNode> create_tensor_node(std::vector<double>& data, const std::vector<int>& shape, const std::vector<int>& stride) {
	
	std::shared_ptr<TensorNode> node = std::make_shared<TensorNode>(data, shape, stride);
	return node;
}

Tensor create_tensor_scalar(const std::vector<int>& shape, double scalar) {
	int N = 1;
	for (int i=0; i<shape.size(); i++) N *= shape[i];
	std::vector<double> data(N, scalar);
	std::shared_ptr<TensorNode> node = create_tensor_node(data, shape);
	return Tensor(node);
}

// creates and returns a tensor. initializes is with 0
Tensor create_tensor_zeros(const std::vector<int>& shape) {
	return create_tensor_scalar(shape, 0.0);	
}

Tensor create_tensor_random(const std::vector<int>& shape, double DIST_MIN, double DIST_MAX) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dist(DIST_MIN, DIST_MAX);

	Tensor x = create_tensor_zeros(shape);
	for (int i=0; i<x.tensor_node->data.size(); i++) {
		x.tensor_node->data[i] = dist(gen);
	}
	return x;
}


Tensor create_tensor_clone_scalar(const Tensor& t, double scalar) {
	
	int N = 1;
	for (int i=0; i<t.shape().size(); i++) N *= t.shape()[i];
	std::vector<double> data(N, scalar);
	std::shared_ptr<TensorNode> tensor_node = std::make_shared<TensorNode>(data, t.shape(), t.stride());
	return tensor_node;
}

// creates it fromt he local graph, with the given data
// should ALSO set the predecessors, sets {a, b, ...} as the predecessors of this node

std::shared_ptr<TensorNode> make_operator_output_node (
	std::vector<double>& data,
	const std::vector<int>& shape,
	const std::vector<std::shared_ptr<TensorNode>>& predecessors,
	std::function<void(TensorNode*)> backward_fn) {
	
	std::shared_ptr<TensorNode> node = create_tensor_node(data, shape);
	for (const auto& predecessor : predecessors) predecessor->indegree++;
	node->predecessors = predecessors;
	node->backward_fn = backward_fn;
	return node;
}


int Tensor::linearize_index(const std::vector<int>& index) {
	return tensor_node->linearize_index(index);
}

// ADD 
Tensor Tensor::operator+(const Tensor& other) const {
	check_shapes_are_same(other);
	std::vector<double> output(tensor_node->data.size());
	for (int i=0; i<tensor_node->data.size(); i++) output[i] = tensor_node->data[i] + other.tensor_node->data[i];
	std::shared_ptr<TensorNode> node = make_operator_output_node(
		output, shape(), {tensor_node, other.tensor_node}, add_backward);
	return Tensor(node);
}


// SUBTRACT
Tensor Tensor::operator-(const Tensor& other) const {
	check_shapes_are_same(other);
	std::vector<double> output(tensor_node->data.size());
	for (int i=0; i<tensor_node->data.size(); i++) output[i] = tensor_node->data[i] - other.tensor_node->data[i];
	std::shared_ptr<TensorNode> node = make_operator_output_node(
		output, shape(), {tensor_node, other.tensor_node}, sub_backward);
	return Tensor(node);
}


// MULTIPLY
Tensor Tensor::operator*(const Tensor& other) const {
	check_shapes_are_same(other);
	std::vector<double> output(tensor_node->data.size());
	for (int i=0; i<tensor_node->data.size(); i++) output[i] = tensor_node->data[i] * other.tensor_node->data[i];
	std::shared_ptr<TensorNode> node = make_operator_output_node(
		output, shape(), {tensor_node, other.tensor_node}, mult_backward);
	return Tensor(node);
}



// DIVIDE
Tensor Tensor::operator/(const Tensor& other) const {
	check_shapes_are_same(other);
	std::vector<double> output(tensor_node->data.size());
	for (int i=0; i<tensor_node->data.size(); i++) output[i] = tensor_node->data[i] / other.tensor_node->data[i];
	std::shared_ptr<TensorNode> node = make_operator_output_node(
		output, shape(), {tensor_node, other.tensor_node}, div_backward);
	return Tensor(node);
}


// MATMUL 2D
Tensor Tensor::MATMUL_2D_ADD(const Tensor& other) const {
	// we have to verify the shapes are the same, and that they are both only 2d
	if (tensor_node->dim() < 2 || other.tensor_node->dim() < 2) throw InvalidTensorShape();
	int INPUT_N, INPUT_M, OTHER_N, OTHER_M, OUTPUT_N, OUTPUT_M;
	
	int dim_A = dim();
	int index_N = dim_A - 2;
	int index_M = dim_A - 1;

	INPUT_N = tensor_node->shape[index_N];
	INPUT_M = tensor_node->shape[index_M];
	OTHER_N = other.tensor_node->shape[0];
	OTHER_M = other.tensor_node->shape[1];
	OUTPUT_N = INPUT_N;
	OUTPUT_M = OTHER_M;

	// now check that the shape is correct for matmul
	if (INPUT_M != OTHER_N) throw InvalidTensorShape();

	// conduct the full matmul operation with the internal data
	// but what is the backward rule if we have all the pointers? 
	// MATMUL should produce one node...
	// we have to update the gradients of the children unanimously... i guess its fine? 
	
	// we have shape N... we can do this by calculating the stride

	// create ouptut buffer
		
	// create the output buffer but we dont really have to do check anything, right? 
	// like realistically for now we can just create it with the B * N * M
	// and delegate the error handling into the otuput layer
	// but we should probably catch that the shapes match for MATMUL
	
	int total_first_dims = 1;
	std::vector<int> output_shape;
	for (int i=0; i<dim_A-2; i++) {
		total_first_dims *= tensor_node->shape[i];
		output_shape.push_back(tensor_node->shape[i]);
	}
	output_shape.push_back(OUTPUT_N);
	output_shape.push_back(OUTPUT_M);
	std::vector<int> output_stride = stride_from_shape(output_shape);

	std::vector<double> output(total_first_dims * OUTPUT_N * OUTPUT_M, 0.0);
		

	// use batched GEMM kernel
	// this is bit inefficient but small can fix later
	BATCHED_GEMM_2D_ADD (
		tensor_node->data,
		tensor_node->stride,
		tensor_node->shape,

		other.tensor_node->data,
		other.tensor_node->stride,
		other.tensor_node->shape,

		output,
		output_stride,
		output_shape
	);
	
	std::shared_ptr<TensorNode> node = make_operator_output_node(
		output, output_shape, {tensor_node, other.tensor_node}, MATMUL_2D_backward);
	return Tensor(node);

}

// ok get the N and M of the current
Tensor Tensor::BIAS_ADD_2D_1D(const Tensor& bias) const {
	
	int dim = tensor_node->dim();
	int index_N = dim-2;
	int index_M = dim-1;

	int AN = shape()[index_N];
	int AM = shape()[index_M];

	// of the output and the input
	// bias is unbatched
	int BM = bias.shape()[0];
	if (AM != BM) throw InvalidTensorShape();

	int total_first_dims = 1;
	for (int i=0; i<index_N; i++) {
		total_first_dims *= tensor_node->shape[i];
	}	
	std::vector<double> output(total_first_dims*AN*AM, 0.0);
	

	// this underlying kernel shoudl work with both 2D and 3D, 
	// we can probably dispatch within that... but i think this should automatically do it
	BATCHED_MAT2D_1D_ADD (
		tensor_node->data,
		stride(),
		shape(),

		bias.tensor_node->data,
		bias.stride(),
		bias.shape(),

		output,
		stride(),
		shape()
	);

	std::shared_ptr<TensorNode> node = make_operator_output_node(
		output, shape(), {tensor_node, bias.tensor_node}, BIAS_ADD_2D_1D_backward);
	return Tensor(node); 

}

/*
Tensor Tensor::pow(double scalar) {
	Tensor new_tensor = create_tensor_clone_scalar(*this, graph, scalar);
	// create the new tensor.. now we can safely do a pointwise pow
	
	std::vector<double> output(tensor_node->data.size());
	for (int i=0; i<tensor_node->data.size(); i++) {
		double new_value = std::pow(tensor_node->data[i], scalar);
		output[i] = new_value;
	}
	TensorNode* raw = make_operator_output_node(output, tensor_node->shape, Op::POW, tensor_node, new_tensor.tensor_node, graph);
	return Tensor(raw, graph);
}
*/


void Tensor::backward() {

	// set the current gradient of the node to 1
	for (int i=0; i<tensor_node->grad.size(); i++) {
		tensor_node->grad[i] = 1.0;
	}

	// make the queue here
	std::deque<std::shared_ptr<TensorNode>> q;
	q.push_back(tensor_node);
	while (q.size() > 0) {

		// get the predecessors, then decrement their indegree
		// if their indegree is 0 and a backward fn is attached, we can add to the queue
		// well everything should be a binary operator
		// oh we can iterate over the parents too, right?
		// call backward on this node, then add the next nodes into the queue one by one

		std::shared_ptr<TensorNode> node = q.front();
		q.pop_front();

		std::cout << "backward() - processing node: " << node.get() << " preds.size(): " << node->predecessors.size()
							<< " backward_fn is null: " << (node->backward_fn == nullptr) << std::endl;
		for (int i=0; i<node->predecessors.size(); i++) {
			std::cout << "  pred[" << i << "]: " << node->predecessors[i].get() << std::endl;
		}

		// call the backward_fn of the node here
		node->backward_fn(node.get());
		for (const auto& pred : node->predecessors) {
			if (pred == nullptr) continue;
			pred->indegree--;
			// add to the backprop queue if the conditions are met
			if (pred->indegree == 0 && pred->backward_fn != nullptr) q.push_back(pred); 
		}
		// we have already pushed into the deque, which has a reference to it.
		// now we can do clear
		node->predecessors.clear();

	}

}
