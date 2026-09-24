#pragma once

#include <vector>
#include <functional>
#include <memory>

#include "calc.h"

using ll = long long;


enum class Op {
	NONE,
	ADD,
	SUB,
	MULT,
	DIV,
	POW,
	EXP,
	MATMUL,
	BIAS_ADD,
	OTHER
};

// so I guess we dont have to do this tensor dereferncing all the time

class TensorNode {
public:
	std::vector<double> data;
	std::vector<double> grad;
	std::vector<int> shape;
	std::vector<int> stride;
	std::vector<TensorNode*> predecessors;
	Op op = Op::OTHER;
	ll indegree = 0;

	// backwards function takes in the current TensorNode
	std::function<void(TensorNode*)> backward_fn;
	
	TensorNode(std::vector<double> _data, std::vector<int> _shape) : data(std::move(_data)), grad(data.size(), 0.0), shape(_shape), stride(_shape.size(), 1) {
		for (int i=shape.size()-2; i>=0; i--) stride[i] = shape[i+1] * stride[i+1];	
	}

	TensorNode(std::vector<double> _data, std::vector<int> _shape, std::vector<int> _stride) :
		data(std::move(_data)),
		grad(data.size(), 0.0),
		shape(_shape),
		stride(_stride) {}
	
	TensorNode(std::vector<double> _data, std::vector<int> _shape, Op _op)
		: TensorNode(std::move(_data), std::move(_shape)) {
		op = _op;
	}
	
	void increase_indegree();
	int dim() {
		return shape.size();
	}

	bool check_shapes_are_same(TensorNode* other) {
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
	void set_predecessors(std::vector<TensorNode*> _predecessors) {
		predecessors = std::move(_predecessors);
	}

};

class Graph {
public:
	std::vector<std::unique_ptr<TensorNode>> nodes;

	Graph() {};
	
	TensorNode* make_node(std::vector<double> data, std::vector<int> shape);
	TensorNode* make_node(std::vector<double> data, std::vector<int> shape, Op op);
	TensorNode* make_node(std::vector<double> data, std::vector<int> shape, std::vector<int> stride);	

	void clear() {nodes.clear();}

};


class Tensor {
private:
	
	// stores the object raw pointers
	std::vector<TensorNode*> get_predecessors(const Tensor& other) {
		return {this->tensor_node, other.tensor_node};
	}

	std::vector<TensorNode*> get_predecessors() {
		return {this->tensor_node};
	}

	
	bool check_shapes_are_same(const Tensor& other) {
		return tensor_node->check_shapes_are_same(other.tensor_node);
	}

public:
	
	// linearly allocate it, and make it be a rectangle in N-D space
	// then we can easily calculate it with the offsets... 
	// and the grad will be the same size as the actual tensor itself, I guess
	TensorNode* tensor_node;
	
	Graph* graph;
	// we can write these constructors. but how do we read data from them
	// its fine, we can probably write some code to create some data loaders, right?
	
	Tensor(TensorNode* _tensor_node, Graph* _graph) : tensor_node(_tensor_node), graph(_graph) {};

	int linearize_index(const std::vector<int>& index);

	// soo what does the operator return? check if the tensor itself holds a graph object
	
	void on_operator(Tensor other);
	
	Tensor operator+(Tensor other);

	Tensor operator-(Tensor other);
	
	Tensor operator*(Tensor other);

	Tensor operator/(Tensor other);
	
	Tensor MATMUL_2D_ADD(Tensor other);
	Tensor BIAS_ADD_2D_1D(Tensor bias);
	
	// return cur ^ scalar
	// creates automatically a tensor node for this scalar value since we are
	// doing a pointwise power 
	Tensor pow(double scalar);
	
	// returns e^ cur
	Tensor exp();

	void backward();

	std::vector<int> shape() {return tensor_node->shape;}
	std::vector<int> stride() {return tensor_node->stride;}
	int dim() {return tensor_node->dim();}
	
	void backward();

};

TensorNode* make_operator_output_node(
	std::vector<double> data, 
	std::vector<int> shape, 
	Op op, 
	TensorNode* a, 
	TensorNode* b, 
	Graph* graph,
	std::function<void(TensorNode*)> backward_fn);

Tensor create_tensor_zeros(std::vector<int> shape, Graph* graph);

Tensor create_tensor_random(std::vector<int> shape, Graph* graph, double DIST_MIN, double DIST_MAX);
Tensor create_tensor_scalar(std::vector<int> shape, Graph* graph, double scalar);

Tensor create_tensor_clone_scalar(Tensor t, Graph* graph, double scalar);
