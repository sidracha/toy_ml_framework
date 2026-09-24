#include "tensor.h"
#include "calc.h"

#include <vector>
#include <deque>
#include <stdexcept>
#include <cmath>
#include <random>

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
	std::vector<int> shape_after;
	std::vector<int> stride_after;
	for (int i=0; i<index_after.size(); i++) {
		shape_after[index_after[i]] = shape[i];
		stride_after[index_after[i]] = stride[i];
	}
	stride = stride_after;
	shape = shape_after;
}

// creates and returns a tensor. initializes is with 0
Tensor create_tensor_zeros(std::vector<int> shape, Graph* graph) {
	int N = 1;
	for (int i=0; i<shape.size(); i++) N *= shape[i];
	std::vector<double> data(N, 0.0);
	TensorNode* raw = graph->make_node(data, shape);

	return Tensor(raw, graph);
}

Tensor create_tensor_random(std::vector<int> shape, Graph* graph, double DIST_MIN, double DIST_MAX) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dist(DIST_MIN, DIST_MAX);

	Tensor x = create_tensor_zeros(shape, graph);
	for (int i=0; i<x.tensor_node->data.size(); i++) {
		x.tensor_node->data[i] = dist(gen);
	}
	return x;
}

Tensor create_tensor_scalar(std::vector<int> shape, Graph* graph, double scalar) {
	Tensor x = create_tensor_zeros(shape, graph);
	for (int i=0; i<x.tensor_node->data.size(); i++) x.tensor_node->data[i] = scalar;
	return x;
} 

Tensor create_tensor_clone_scalar(Tensor t, Graph* graph, double scalar) {
	int N = 1;
	std::vector<int> shape = t.tensor_node->shape;
	std::vector<int> stride = t.tensor_node->stride;
	for (int i=0; i<shape.size(); i++) N *= shape[i];
	std::vector<double> data(N, 0.0);
	TensorNode* raw = graph->make_node(data, shape, stride);
	return Tensor(raw, graph);
}

TensorNode* Graph::make_node(std::vector<double> data, std::vector<int> shape) {
	auto node = std::make_unique<TensorNode>(std::move(data), shape);
	TensorNode* raw = node.get();
	nodes.push_back(std::move(node));
	return raw;
}

TensorNode* Graph::make_node(std::vector<double> data, std::vector<int> shape, std::vector<int> stride) {
	auto node = std::make_unique<TensorNode>(std::move(data), shape);
	TensorNode* raw = node.get();
	nodes.push_back(std::move(node));
	return raw;
}

TensorNode* Graph::make_node(std::vector<double> data, std::vector<int> shape, Op op) {
	auto node = std::make_unique<TensorNode>(std::move(data), shape, op);
	TensorNode* raw = node.get();
	nodes.push_back(std::move(node));
	return raw;
}

// creates it fromt he local graph, with the given data
// should ALSO set the predecessors, sets {a, b} as the predecessors of this node
TensorNode* make_operator_output_node(std::vector<double> data, std::vector<int> shape, Op op, TensorNode* a, TensorNode* b, Graph* graph) {
	if (graph == nullptr) throw std::runtime_error("Graph must be valid for all operations");
	TensorNode* raw = graph->make_node(data, shape, op);
	raw->set_predecessors({a, b});
	return raw;	
}


int Tensor::linearize_index(const std::vector<int>& index) {
	return tensor_node->linearize_index(index);
}

// ADD 
Tensor Tensor::operator+(Tensor other) {
	check_shapes_are_same(other);
	std::vector<double> output(tensor_node->data.size());
	for (int i=0; i<tensor_node->data.size(); i++) output[i] = tensor_node->data[i] + other.tensor_node->data[i];
	TensorNode* raw = make_operator_output_node(output, tensor_node->shape, Op::ADD, tensor_node, other.tensor_node, graph); 
	return Tensor(raw, graph);
}


// SUBTRACT 
Tensor Tensor::operator-(Tensor other) {
	check_shapes_are_same(other);
	std::vector<double> output(tensor_node->data.size());
	for (int i=0; i<tensor_node->data.size(); i++) output[i] = tensor_node->data[i] - other.tensor_node->data[i];
	TensorNode* raw = make_operator_output_node(output, tensor_node->shape, Op::SUB, tensor_node, other.tensor_node, graph); 
	return Tensor(raw, graph);
}


// MULTIPLY
Tensor Tensor::operator*(Tensor other) {
	check_shapes_are_same(other);
	std::vector<double> output(tensor_node->data.size());
	for (int i=0; i<tensor_node->data.size(); i++) output[i] = tensor_node->data[i] * other.tensor_node->data[i];
	TensorNode* raw = make_operator_output_node(output, tensor_node->shape, Op::MULT, tensor_node, other.tensor_node, graph); 
	return Tensor(raw, graph);
}



// DIVIDE 
Tensor Tensor::operator/(Tensor other) {
	check_shapes_are_same(other);
	std::vector<double> output(tensor_node->data.size());
	for (int i=0; i<tensor_node->data.size(); i++) output[i] = tensor_node->data[i] / other.tensor_node->data[i];
	TensorNode* raw = make_operator_output_node(output, tensor_node->shape, Op::DIV, tensor_node, other.tensor_node, graph); 
	return Tensor(raw, graph);

}


// MATMUL 2D
Tensor Tensor::MATMUL_2D_ADD(Tensor other) {
	// we have to verify the shapes are the same, and that they are both only 2d
	if (other.tensor_node->dim() != 2 || tensor_node->dim() != 2) throw InvalidTensorShape();
	
	int INPUT_N = tensor_node->shape[0];
	int INPUT_M = tensor_node->shape[1];
	int OTHER_N = other.tensor_node->shape[0];
	int OTHER_M = other.tensor_node->shape[1];
	int OUTPUT_N = INPUT_M;
	int OUTPUT_M = OTHER_N;

	// now check that the shape is correct for matmul
	if (INPUT_M != OTHER_N) throw InvalidTensorShape();

	// conduct the full matmul operation with the internal data
	// but what is the backward rule if we have all the pointers? 
	// MATMUL should produce one node...
	// we have to update the gradients of the children unanimously... i guess its fine? 
	
	// we have shape N... we can do this by calculating the stride

	// create ouptut buffer
	std::vector<double> output(OUTPUT_N * OUTPUT_M, 0.0);
		

	// use 2D GEMM KERNEL
	std::vector<int> output_shape = {OUTPUT_N, OUTPUT_M};
	std::vector<int> output_stride = {OUTPUT_M, 1};
	GEMM_2D_ADD (
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
	
	return Tensor(make_operator_output_node(output, output_shape, Op::MATMUL, tensor_node, other.tensor_node, graph), graph);

}

// ok get the N and M of the current
Tensor Tensor::BIAS_ADD_2D_1D(Tensor bias) {

	int AN = shape()[0];
	int AM = shape()[1];
	
	int BM = bias.shape()[0];
	if (AM != BM) throw InvalidTensorShape();

	std::vector<double> output(AN*AM, 0.0);

	MAT2D_1D_ADD (
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

	return Tensor(make_operator_output_node(output, shape(), Op::BIAS_ADD, tensor_node, bias.tensor_node, graph), graph); 

}

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


