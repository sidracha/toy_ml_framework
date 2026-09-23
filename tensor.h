#pragma once

#include <vector>

using ll = long long;

enum class Op {
	NONE,
	ADD,
	SUB,
	MULT,
	DIV,
	POW,
	EXP,
	OTHER
};

// so I guess we dont have to do this tensor dereferncing all the time


class TensorNode {
public:
	std::vector<double> data;
	std::vector<double> grad;
	std::vector<int> shape;
	std::vector<int> stride;
	Op op = OTHER;
	ll indegree = 0; 
	
	TensorNode(std::vector<double> _data, std::vector<int> _shape) : data(std::move(_data)), grad(data.size(), 0.0), shape(_shape), stride(_shape.size(), 1) {
		for (int i=shape.size()-1, i>=0; i--) stride[i] = shape[i+1] * shape[i];	
	};
	
	TensorNode(std::vector<double> _data, std::vector<int> _shape, Op _op) {
		TensorNode(_data, _shape);
		op = _op;
	}
	
	void increase_indegree();

};

class Graph {
	std::vector<std::unique_ptr<TensorNode>> nodes;

	Graph() {};
	
	TensorNode* make_node(vector<double> data, vector<int> size);
	TensorNode* make_node(vector<double> data, vector<int> size, Op op);

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
		return (other.tensor_node->size == this->tensor_node->shape);
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

	int linearize_index(vector<int> index);

	// soo what does the operator return? check if the tensor itself holds a graph object
	
	Tensor operator+(Tensor other);
	Tensor operator+(double other);

	Tensor operator-(Tensor other);
	Tensor operator-(double other);
	
	Tensor operator*(Tensor other);
	Tensor operator*(double other);

	Tensor operator/(Tensor other);
	Tensor operator/(double other);
	
	// return cur ^ other
	Tensor pow(double other);
	
	// returns e^ cur
	Tensor exp();

	void backward();

};
