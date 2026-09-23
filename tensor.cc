#include "tensor.h"

#include <vector>
#include <deque>
#include <stdexcept>
#include <cmath>

class TensorShapeMismatch : public std::runtime_error {
public:
	TensorShapeMismatch(Tensor a, Tensor b) : std::runtime_error("Tensor shapes are mismatched") {};
}

class InvalidTensorShape : public std::runtime_error {
public:
	InvalidTensorShape() : std::runtime_error("Invalid Tensor Shape") {};
}

void TensorNode::increase_indegree() {
	indegree++;
}


// formula stride[0] * index[0] + stride[1] * index[1] ... stride[n-1] * index[n-1];
int TensorNode::linearize_index(const std::vector<int>& index) {
	int ret = 0;
	for (int i=0; i<stride.size(); i++) {
		ret += stride[i] * index[i];
	}
	return ret;
}

TensorNode* Graph::make_node(std::vector<double> data, std::vector<int> shape) {
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
TensorNode* make_operator_output_node(std::vector<double> data, vector<int> shape, Op op) {
	if (graph == nullptr) throw std::runtime_error("Graph must be valid for all operations");
	TensorNode* raw = graph->make_node(data, shape, Op);
	return raw;	
}


int Tensor::linearize_index(const vector<int>& index) {
	return tensor_node->linearize_index(index);
}

// ADD 
Tensor Tensor::operator+(Tensor other) {
	check_shapes_are_same(other);
	// assert that the shapes and the data sizes are the same
	std::vector<double> output(tensor_node->data.size());
	for (int i=0; i<data.size(); i++) output[i] = tensor_node->data[i] + other.tensor_node->data[i];
	TensorNode* raw = make_operator_output_node(output, tensor_node->shape, ADD); 
	return Tensor(raw, graph);
}

Tensor Tensor::operator+(double other) {
	std::vector<double> output(tensor_node->data.size());
	for (int i=0; i<data.size(); i++) output[i] = tensor_node->data[i] + other;
	TensorNode* raw = make_operator_output_node(output, tensor_node->shape, ADD); 
	return Tensor(raw, graph);
}


// SUBTRACT 
Tensor Tensor::operator+(Tensor other) {
	check_shapes_are_same(other);
	// assert that the shapes and the data sizes are the same
	std::vector<double> output(tensor_node->data.size());
	for (int i=0; i<data.size(); i++) output[i] = tensor_node->data[i] - other.tensor_node->data[i];
	TensorNode* raw = make_operator_output_node(output, tensor_node->shape, SUB); 
	return Tensor(raw, graph);
}

Tensor Tensor::operator+(double other) {
	std::vector<double> output(tensor_node->data.size());
	for (int i=0; i<data.size(); i++) output[i] = tensor_node->data[i] - other;
	TensorNode* raw = make_operator_output_node(output, tensor_node->shape, SUB); 
	return Tensor(raw, graph);
}


// MULTIPLY
Tensor Tensor::operator+(Tensor other) {
	check_shapes_are_same(other);
	// assert that the shapes and the data sizes are the same
	std::vector<double> output(tensor_node->data.size());
	for (int i=0; i<data.size(); i++) output[i] = tensor_node->data[i] * other.tensor_node->data[i];
	TensorNode* raw = make_operator_output_node(output, tensor_node->shape, MULT); 
	return Tensor(raw, graph);
}

Tensor Tensor::operator+(double other) {
	std::vector<double> output(tensor_node->data.size());
	for (int i=0; i<data.size(); i++) output[i] = tensor_node->data[i] * other;
	TensorNode* raw = make_operator_output_node(output, tensor_node->shape, MULT); 
	return Tensor(raw, graph);
}


// DIVIDE 
Tensor Tensor::operator+(Tensor other) {
	check_shapes_are_same(other);
	// assert that the shapes and the data sizes are the same
	std::vector<double> output(tensor_node->data.size());
	for (int i=0; i<data.size(); i++) output[i] = tensor_node->data[i] / other.tensor_node->data[i];
	TensorNode* raw = make_operator_output_node(output, tensor_node->shape, DIV); 
	return Tensor(raw, graph);
}

Tensor Tensor::operator+(double other) {
	std::vector<double> output(tensor_node->data.size());
	for (int i=0; i<data.size(); i++) output[i] = tensor_node->data[i] / other;
	TensorNode* raw = make_operator_output_node(output, tensor_node->shape, DIV); 
	return Tensor(raw, graph);
}

// MATMUL 2D
Tensor Tensor::MATMUL_2D(Tensor other) {
	// we have to verify the shapes are the same, and that they are both only 2d
	if (other.tensor_node->shape.size() != 2 || tensor_node->shape.size() != 2) throw InvalidShapeError();
	
	int INPUT_N = tensor_node->shape[0];
	int INPUT_M = tensor_node->shape[1];
	int OTHER_N = other.tensor_node->shape[0];
	int OTHER_M = other.tensor_node->shape[1];
	int OUTPUT_N = INPUT_M;
	int OUTPUT_M = OTHER_N;

	// now check that the shape is correct for matmul
	if (INPUT_M != OTHER_N) throw InvalidShapeErrpr();

	// conduct the full matmul operation with the internal data
	// but what is the backward rule if we have all the pointers? 
	// MATMUL should produce one node...
	// we have to update the gradients of the children unanimously... i guess its fine? 
	
	// we have shape N... we can do this by calculating the stride
	
	//create output data.. it size is N*M
	vector<double> output(N*M);
	
	// iterate over output positions
	for (int i=0; i<OUTPUT_N; i++) {
		for (int j=0; j<OUTPUT_M) {
			// calculate the dot product between the rows and columns
			// use row i for input and col j for output and iterate through k, which should be INPUT_M == OTHER_N sized
			
			double dot_product = 0;
			for (int k=0; k<INPUT_M; k++) {
				// we resolve both of the indexes...
				int A_row = i;
				int A_col = k;
				
				int B_row = k;
				int B_col = j;
				
				// now turn into a linearized index
				int A_linearized_index = linearize_index({i, k});
				int B_linearized_index = linearize_index({k, j});

				dot_product += tensor_node->data[A_linearized_index] + other.tensor_node->data[B_linearized_index];
			}
			output[i][j] = dot_product;

		}
	}
	vector<int> shape = {OUTPUT_N, OUTPUT_M};
	return make_operator_output_node(output, shape, MATMUL);

}

// the root of the backward pass. we want to do this in reverse topological order
// we also need to calculate the indegree, right? 
// REVERSE topological order,
// so we increase the indegree of the nodes that caused the result
// increase the indegre of the nodes

// start at the current node, get the 2 parents and use a queue
void Tensor::backward() {
	

	// can we fold it into mult and into that? probably not... better to be extendable I guess.....
	std::deque<Tensor*> q;
	// just add the tensor, and we have all the information that we need... and we can modify the gradient, THEN add it to the queue...
	this->grad = 1;
	q.push_back(this);

	while (q.size() > 0) {
		// get the next node, modify the grad and the indegree of the CHILDREN, then if the indegree is 0, we add to the queue
		Tensor* node = q.front();
		q.pop_front();
		
		Tensor* a = nullptr;
		Tensor* b = nullptr;

		if (node->predecessors.size() <= 0) continue;
		if (node->predecessors.size() == 1) {
			a = node->predecessors[0];
		}
		else {
			a = node->predecessors[0];
			b = node->predecessors[1];
		}
		
		if (node->op == Op::ADD) {
			// we can just add the gradient, since the gradient carries over I guess
			a->grad += node->grad;
			b->grad += node->grad;
		} 

		else if (node->op == Op::SUB) {
			a->grad += node->grad;
			b->grad -= node->grad;
		} 
 
		else if (node->op == Op::DIV) {
			a->grad += (node->grad / b->value);
			double inverse_b_squared = 1 / (b->value * b->value);
			b->grad += (node->grad * -a->value * inverse_b_squared);
		}

		else if (node->op == Op::MULT) {
			a->grad += (node->grad * b->value);
			b->grad += (node->grad * a->value);
		}
		
		else if (node->op == Op::POW) {
			double dda = b->value * (std::pow(a->value, b->value-1));
			double ddb = node->value * log(b->value);
			a->grad += (node->grad * dda);
			b->grad += (node->grad * ddb);
		}

		else if (node->op == Op::EXP) {
			a->grad += (node->grad * node->value);
		}

		else throw std::runtime_error("Op is not valid");
	

		for (Tensor* predecessor : node->predecessors) {
			
			if (--predecessor->indegree == 0) q.push_back(predecessor); 
			
		}

	}

}
