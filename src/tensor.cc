#include "tensor.h"
#include "calc.h"
#include "tensor_grad.h"
#include "utils.h"

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
	node->predecessors = predecessors;
	node->backward_fn = backward_fn;
	return node;
}

std::shared_ptr<TensorNode> make_operator_output_node (
	std::vector<double>& data,
	const std::vector<int>& shape,
	const std::vector<int>& stride,
	const std::vector<std::shared_ptr<TensorNode>>& predecessors,
	std::function<void(TensorNode*)> backward_fn) {
	
	std::shared_ptr<TensorNode> node = create_tensor_node(data, shape, stride);
	node->predecessors = predecessors;
	node->backward_fn = backward_fn;
	return node;
}

int Tensor::linearize_index(const std::vector<int>& index) {
	return tensor_node->linearize_index(index);
}

// ADD with broadcasting (this is always the bigger tensor)
Tensor Tensor::operator+(const Tensor& other) const {
	std::vector<int> shape_a = shape();
	std::vector<int> shape_b = other.shape();
	std::vector<int> stride_a = stride();
	std::vector<int> stride_b = other.stride();

	int dim_diff = shape_a.size() - shape_b.size();
	for (int i = 0; i < dim_diff; i++) { shape_b.insert(shape_b.begin(), 1); stride_b.insert(stride_b.begin(), 0); }

	std::vector<double> output(tensor_node->data.size());
	std::vector<int> odometer(shape_a.size(), 0);
	int odometer_index = 0;

	while (odometer_index >= 0) {
		int other_index = calculate_offset_broadcast(odometer, stride_b, shape_b);
		output[odometer_index] = tensor_node->data[odometer_index] + other.tensor_node->data[other_index];
		odometer_index = odometer_next(odometer, shape_a, stride_a);
	}

	std::shared_ptr<TensorNode> node = make_operator_output_node(
		output, shape_a, {tensor_node, other.tensor_node}, add_backward);
	return Tensor(node);
}

Tensor Tensor::operator+(double scalar) const {
	
	int N = tensor_node->shape[dim()-1];
	std::vector<double> scalar_vector(N, scalar);
	std::shared_ptr<TensorNode> scalar_node = make_operator_output_node(
		scalar_vector, {N}, {}, nullptr);
	
	return operator+(Tensor(scalar_node));
}


// SUBTRACT
Tensor Tensor::operator-(const Tensor& other) const {
	std::vector<int> shape_a = shape();
	std::vector<int> shape_b = other.shape();
	std::vector<int> stride_a = stride();
	std::vector<int> stride_b = other.stride();

	int dim_diff = shape_a.size() - shape_b.size();
	for (int i = 0; i < dim_diff; i++) { shape_b.insert(shape_b.begin(), 1); stride_b.insert(stride_b.begin(), 0); }

	std::vector<double> output(tensor_node->data.size());
	std::vector<int> odometer(shape_a.size(), 0);
	int odometer_index = 0;

	while (odometer_index >= 0) {
		int other_index = calculate_offset_broadcast(odometer, stride_b, shape_b);
		output[odometer_index] = tensor_node->data[odometer_index] - other.tensor_node->data[other_index];
		odometer_index = odometer_next(odometer, shape_a, stride_a);
	}

	std::shared_ptr<TensorNode> node = make_operator_output_node(
		output, shape_a, {tensor_node, other.tensor_node}, sub_backward);
	return Tensor(node);
}

Tensor Tensor::operator-(double scalar) const {
	
	int N = tensor_node->shape[dim()-1];
	std::vector<double> scalar_vector(N, scalar);
	std::shared_ptr<TensorNode> scalar_node = make_operator_output_node(
		scalar_vector, {N}, {}, nullptr);
	
	return operator-(Tensor(scalar_node));
}


// MULTIPLY

Tensor Tensor::operator*(const Tensor& other) const {
	std::vector<int> shape_a = shape();
	std::vector<int> shape_b = other.shape();
	std::vector<int> stride_a = stride();
	std::vector<int> stride_b = other.stride();

	int dim_diff = shape_a.size() - shape_b.size();
	for (int i = 0; i < dim_diff; i++) { shape_b.insert(shape_b.begin(), 1); stride_b.insert(stride_b.begin(), 0); }

	std::vector<double> output(tensor_node->data.size());
	std::vector<int> odometer(shape_a.size(), 0);
	int odometer_index = 0;

	while (odometer_index >= 0) {
		int other_index = calculate_offset_broadcast(odometer, stride_b, shape_b);
		output[odometer_index] = tensor_node->data[odometer_index] * other.tensor_node->data[other_index];
		odometer_index = odometer_next(odometer, shape_a, stride_a);
	}

	std::shared_ptr<TensorNode> node = make_operator_output_node(
		output, shape_a, {tensor_node, other.tensor_node}, mult_backward);
	return Tensor(node);
}

Tensor Tensor::operator*(double scalar) const {
	
	int N = tensor_node->shape[dim()-1];
	std::vector<double> scalar_vector(N, scalar);
	std::shared_ptr<TensorNode> scalar_node = make_operator_output_node(
		scalar_vector, {N}, {}, nullptr);
	
	return operator*(Tensor(scalar_node));
}


// DIVIDE
Tensor Tensor::operator/(const Tensor& other) const {
	std::vector<int> shape_a = shape();
	std::vector<int> shape_b = other.shape();
	std::vector<int> stride_a = stride();
	std::vector<int> stride_b = other.stride();

	int dim_diff = shape_a.size() - shape_b.size();
	for (int i = 0; i < dim_diff; i++) { shape_b.insert(shape_b.begin(), 1); stride_b.insert(stride_b.begin(), 0); }

	std::vector<double> output(tensor_node->data.size());
	std::vector<int> odometer(shape_a.size(), 0);
	int odometer_index = 0;

	while (odometer_index >= 0) {
		int other_index = calculate_offset_broadcast(odometer, stride_b, shape_b);
		output[odometer_index] = tensor_node->data[odometer_index] / other.tensor_node->data[other_index];
		odometer_index = odometer_next(odometer, shape_a, stride_a);
	}

	std::shared_ptr<TensorNode> node = make_operator_output_node(
		output, shape_a, {tensor_node, other.tensor_node}, div_backward);
	return Tensor(node);
}

Tensor Tensor::operator/(double scalar) const {
	
	int N = tensor_node->shape[dim()-1];
	std::vector<double> scalar_vector(N, scalar);
	std::shared_ptr<TensorNode> scalar_node = make_operator_output_node(
		scalar_vector, {N}, {}, nullptr);
	
	return operator/(Tensor(scalar_node));
}

Tensor Tensor::pow(double scalar) const {
	
	std::vector<double> scalar_vector(1, scalar);
	std::vector<double> output(tensor_node->data.size());
	for (int i=0; i<output.size(); i++) output[i] = std::pow(tensor_node->data[i], scalar);
	
	std::shared_ptr<TensorNode> scalar_node = make_operator_output_node(
		scalar_vector, {1}, {}, pow_backward);

	std::shared_ptr<TensorNode> node = make_operator_output_node(
		output, shape(), {tensor_node, scalar_node}, pow_backward);
	return Tensor(node);

}



// MATMUL 2D
Tensor Tensor::MATMUL_2D(const Tensor& other) const {
	// we have to verify the shapes are the same, and that they are both only 2d
	if (tensor_node->dim() < 2 || other.tensor_node->dim() < 2) throw InvalidTensorShape();
	int INPUT_N, INPUT_M, OTHER_N, OTHER_M, OUTPUT_N, OUTPUT_M;
	
	int dim_A = dim();
	int dim_other = other.dim();
	int index_N = dim_A - 2;
	int index_M = dim_A - 1;

	INPUT_N = tensor_node->shape[index_N];
	INPUT_M = tensor_node->shape[index_M];
	
	OTHER_N = other.tensor_node->shape[dim_other-2];
	OTHER_M = other.tensor_node->shape[dim_other-1];
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

bool bool_odometer_next(std::vector<int>& odometer, const std::vector<int>& shape) {
	if (odometer.size() == 0) return false;

	
	int carry = 0;
	bool first = false;
	for (int i=odometer.size()-1; i>=0; i--) {
		if (!first) { // add one to the first one
			odometer[i]++;
			first = true;
		}
		odometer[i] += carry;
		int dig = (odometer[i] % shape[i]);
		carry = (odometer[i] / shape[i]);
		odometer[i] = dig;
		if (carry == 0) break;
	}
	if (carry > 0) return false;
	return true;
}

int calculate_offset(
		const std::vector<int>& odometer,
		const std::vector<int>& stride) {
	
	int ret_index = 0;
	for (int i=0; i<odometer.size(); i++) {
		ret_index += odometer[i] * stride[i];
	}
	return ret_index;

}

Tensor transpose(const Tensor& t) {
	
	// transpose the last 2 dimensions across the batch
	int dim = t.dim();
	if (dim < 2) throw std::runtime_error("transpose called on dim < 2");
	int index_N = dim - 2;
	int index_M = dim - 1;
		
	std::vector<int> odometer;
	std::vector<int> output_shape;
	for (int i=0; i<dim-2; i++) {
		odometer.push_back(0);
		output_shape.push_back(t.tensor_node->shape[i]);
	}
	
	int input_N = t.tensor_node->shape[index_N];
	int input_M = t.tensor_node->shape[index_M];
	int output_N = input_M;
	int output_M = input_N;
	output_shape.push_back(output_N);
	output_shape.push_back(output_M);
	std::vector<int> output_stride(dim);
	output_stride[dim-1] = 1;
	for (int i=dim-2; i>=0; i--) output_stride[i] = output_stride[i+1] * output_shape[i+1]; 
		
	
	std::vector<double> output(t.tensor_node->data.size());
	
	// keep going as long as the odometers are the same...
	// should be the same since its the same basically.. yeah
	// we have to add the same amounts so whatever
	int exists_next = true;
	while (exists_next) {
		// now iterate through all of the indexes 
		// of the last 2 indexes...
		// and we calculate it
		int offset = calculate_offset(odometer, t.tensor_node->stride);
		for (int i=0; i<input_N; i++) {
			for (int j=0; j<input_M; j++) {
				int input_index = offset + i * t.tensor_node->stride[index_N] + j * t.tensor_node->stride[index_M];
				int output_index = offset + j * output_stride[index_N] + i * output_stride[index_M];

				output[output_index] = t.tensor_node->data[input_index];

			}
		}
		std::vector<int> t_shape = t.shape();
		exists_next = bool_odometer_next(odometer, t_shape);
	}

	std::shared_ptr<TensorNode> node = make_operator_output_node(
		output, output_shape, {t.tensor_node}, transpose_backward);

	return Tensor(node);
}


void Tensor::backward(bool retain_graph) {

	// should we make the graph here actually? and traverse
	// it in reverse topological order? maybe have like an indegree
	// unordered map or something
	
	// walk backwards, do a dfs? id rather do a bfs to protect the call stack
	std::deque<std::shared_ptr<TensorNode>> indegree_q;
	std::unordered_map<std::shared_ptr<TensorNode>, int> indegree;
	indegree_q.push_back(tensor_node);
	indegree[tensor_node] = 0;
	while (indegree_q.size() > 0) {
		// add the predecessors to the queue, but also track in the indegree set. dont 
		// push back if icurrently already in the indegree set
		std::shared_ptr<TensorNode> node = indegree_q.front();
		indegree_q.pop_front();

		for (const auto& pred : node->predecessors) {
			if (indegree[pred] == 0) { //doenst exist, have to add to the queue
				indegree_q.push_back(pred);
			}
			indegree[pred]++;
		} 

	}
	// now we can add all the 0 indegree ones into
	// the queue

	// set the current gradient of the node to 1
	for (int i=0; i<tensor_node->grad.size(); i++) {
		tensor_node->grad[i] = 1.0;
	}
	
	std::deque<std::shared_ptr<TensorNode>> q;
	for (const auto& [key, val] : indegree) {
		if (val == 0) q.push_back(key);
	} 
	while (q.size() > 0) {

		// get the predecessors, then decrement their indegree
		// if their indegree is 0 and a backward fn is attached, we can add to the queue
		// well everything should be a binary operator
		// oh we can iterate over the parents too, right?
		// call backward on this node, then add the next nodes into the queue one by one

		std::shared_ptr<TensorNode> node = q.front();
		q.pop_front();

		// call the backward_fn of the node here
		node->backward_fn(node.get());
		for (const auto& pred : node->predecessors) {
			if (pred == nullptr) continue;
			indegree[pred]--;
			// add to the backprop queue if the conditions are met
			if (indegree[pred] == 0 && pred->backward_fn != nullptr) q.push_back(pred); 
		}
		// we have already pushed into the deque, which has a reference to it.
		// now we can do clear
		if (!retain_graph) node->predecessors.clear();

	}


}
