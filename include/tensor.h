#pragma once

#include <vector>
#include <functional>
#include <memory>

#include "calc.h"

using ll = long long;


class TensorNode {
public:
	std::vector<double> data;
	std::vector<double> grad;
	std::vector<int> shape;
	std::vector<int> stride;
	std::vector<std::shared_ptr<TensorNode>> predecessors;
	ll indegree = 0;

	// backwards function takes in the current TensorNode
	std::function<void(TensorNode*)> backward_fn;
	
	TensorNode(std::vector<double>& _data, const std::vector<int>& _shape) : data(std::move(_data)), grad(data.size(), 0.0), shape(_shape), stride(_shape.size(), 1) {
		for (int i=shape.size()-2; i>=0; i--) stride[i] = shape[i+1] * stride[i+1];
	}

	TensorNode(std::vector<double>& _data, const std::vector<int>& _shape, const std::vector<int>& _stride) :
		data(std::move(_data)),
		grad(data.size(), 0.0),
		shape(_shape),
		stride(_stride) {}
	
	void increase_indegree();
	int dim() {
		return shape.size();
	}

	bool check_shapes_are_same(const std::shared_ptr<TensorNode>& other) {
		return (this->shape == other->shape && this->stride == other->stride);
	}

	int linearize_index(const std::vector<int>& index) {
		int idx = 0;
		for (int i=0; i<dim(); i++) {
			idx += stride[i] * index[i];
		}
		return idx;
	}
	
	double data_at(const std::vector<int>& index) {
		return data[linearize_index(index)];
	}

	double grad_at(const std::vector<int>& index) {
		return grad[linearize_index(index)];
	}

	void permute(const std::vector<int>& index_after);
	void set_predecessors(std::vector<std::shared_ptr<TensorNode>>& _predecessors) {
		predecessors = std::move(_predecessors);
	}

};


class Tensor {
private:
	
	bool check_shapes_are_same(const Tensor& other) {
		return tensor_node->check_shapes_are_same(other.tensor_node);
	}

public:
	

	std::shared_ptr<TensorNode> tensor_node;
	
		// we can write these constructors. but how do we read data from them
	// its fine, we can probably write some code to create some data loaders, right?
	
	Tensor(std::shared_ptr<TensorNode> _tensor_node) : tensor_node(_tensor_node) {};

	int linearize_index(const std::vector<int>& index);

	// soo what does the operator return? check if the tensor itself holds a graph object
	
	void on_operator(const Tensor& other);
	
	Tensor operator+(const Tensor& other);

	Tensor operator-(const Tensor& other);
	
	Tensor operator*(const Tensor& other);

	Tensor operator/(const Tensor& other);
	
	Tensor MATMUL_2D_ADD(const Tensor& other);
	Tensor BIAS_ADD_2D_1D(const Tensor& bias);
	
	// return cur ^ scalar
	// creates automatically a tensor node for this scalar value since we are
	// doing a pointwise power 
	Tensor pow(double scalar);
	
	// returns e^ cur
	Tensor exp();

	void backward();

	std::vector<int> shape() const {return tensor_node->shape;}
	std::vector<int> stride() const {return tensor_node->stride;}
	int dim() const {return tensor_node->dim();}
	
};

std::shared_ptr<TensorNode> make_operator_output_node(
	std::vector<double>& data,
	const std::vector<int>& shape,
	const std::vector<std::shared_ptr<TensorNode>>& predecessors,
	std::function<void(TensorNode*)> backward_fn);

Tensor create_tensor_zeros(const std::vector<int>& shape);

Tensor create_tensor_random(const std::vector<int>& shape, double DIST_MIN, double DIST_MAX);
Tensor create_tensor_scalar(const std::vector<int>& shape, double scalar);

Tensor create_tensor_clone_scalar(const Tensor& t, double scalar);
