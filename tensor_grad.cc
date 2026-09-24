// The problem:
//
// We need a Tensor class that stores value and gradient
// The tensors that produced it, and enough information to propogate the gradients backwards

// first implement addition and multiplication


// example equation:
// d = a + b*c
// we need to find d/da, d/db, d/dc


// so we have (b*c) -> tensor T
// then a + T -> d


// so the graph goes
//
// we have X = ABC
// then T = AB
// dT/dA = B
// dT/dB = A

// X = TC
//
// dX/dT = git ls-files | xargs wc -lC
// dX/dT



// a node can only have 2 predecessors.
// when we create a node, we want to add to the predecessors of the new node that 
// is created

#include <vector>
#include <deque>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <random>

#include "tensor.h"
#include "nn.h"
#include "losses.h"
#include "linear.h"
#include "calc.h"

class Optimizer {
public:
	std::vector<Linear>& layers;

	Optimizer(std::vector<Linear>& _layers) : layers(_layers) {};
	
	void zero_grad() {
		// set everything in the graph to 0.
		// do we traverse the graph, and then set everything to 0...
		// but all the intemediate ones should also be 0 so its fine...
		for (Linear& layer : layers) {
		

		}
	}

};

// THE BACKWARD PASS FOR EACH OPERATOR ON THE TENSOR
// ASSUMES THAT GRAD FOR THE TENSOR IS CURRENTLY ALREADY CALCULATED
// BACKPROP THE GRADIENTS IN THE CURRENT ONE TO THE CHILDREN

bool verify_predecessor_size(TensorNode* node, int expected) {
	if (node->predecessors.size() != expected) throw std::runtime_error("Predecessor size is not what is expected"); 
}


// ADD HAS TO BE OF THE SAME SHAPE
void add_backward(TensorNode* node) {
	verify_predecessor_size(node, 2);
	
	TensorNode* a = node->predecessors[0];
	TensorNode* b = node->predecessors[1];
	
	for (int i=0; i<node->grad.size(); i++) {
		a->grad[i] += node->grad[i];
		b->grad[i] += node->grad[i];
	}

}

// x = a - b 
// y = f(x)
// dy/da = dx/da * dx/dy
// dy/db = dx/db * dx/dy
// dx/db = -1
void sub_backward(TensorNode* node) {
	verify_predecessor_size(node, 2);
	
	TensorNode* a = node->predecessors[0];
	TensorNode* b = node->predecessors[1];

	for (int i=0; i<node->grad.size(); i++) {
		a->grad[i] += node->grad[i];
		b->grad[i] -= node->grad[i];
	}
}

// x = ab
// y = f(x)
// dy/da = dy/dx * dx/da
// dx = da = b
// so += grad[node] * b

void mult_backward(TensorNode* node) {
	verify_predecessor_size(node, 2);
	
	TensorNode* a = node->predecessors[0];
	TensorNode* b = node->predecessors[1];

	for (int i=0; i<node->grad.size(); i++) {
		a->grad[i] += (node->grad[i] * b->data[i]);
		b->grad[i] += (node->grad[i] * a->data[i]);
	}

}

// x = a/b
// dx/da = 1/b
// dx/db = -a/b^2

void div_backward(TensorNode* node) {
	verify_predecessor_size(node, 2);

	TensorNode* a = node->predecessors[0];
	TensorNode* b = node->predecessors[1];
	
	for (int i=0; i<node->grad.size(); i++) {
		double dda = 1/b->data[i];
		double b_data = b->data[i];
		double ddb = -a->data[i] * (1/(b_data * b_data)); 
		a->grad[i] += (node->grad[i] * dda);
		b->grad[i] += (node->grad[i] * ddb);
		
	}

}

// lets say C = AB for this
// ddA = B * (ddC)^T
// ddB = A^T * (ddC)

void MATMUL_2D_backward(TensorNode* node) {
	verify_predecessor_size(node, 2);
	
	TensorNode* A = node->predecessors[0];
	TensorNode* B = node->predecessors[1];
	
	// ddA, first transpose ddC, current node grad is ddC
	node->permute({1, 0});
	GEMM_2D_ADD (
		B->data,
		B->stride,
		B->shape,
		node->grad,
		node->stride,
		node->shape,
		A->grad,
		A->stride,
		A->shape
	);

	// ddC transpose back 
	node->permute({1, 0});
	
	// transpose A	
	A->permute({1, 0});
	GEMM_2D_ADD (
		A->data,
		A->stride,
		A->shape,
		node->grad,
		node->stride,
		node->shape,
		B->grad,
		B->stride,
		B->shape
	);
	// transpose A back
	A->permute({1, 0});
	
	// the gradients are now populated into the predecessors
}

void BIAS_ADD_2D_1D_backward(TensorNode* node) {
	
	TensorNode* A = node->predecessors[0];
	TensorNode* B = node->predecessors[1];
	
	// A is the input matrix of size N, M
	int N = A->shape[0];
	int M = A->shape[1];
	// B.shape[0] = M
	// basically just add all the contributions since its column wise
	// and for a, the grad is just all 1s, so its fine we just add it 
	for (int j=0; j<M; j++) {
		double ddb = 0.0;
		for (int i=0; i<N; i++) {
			int node_index = node->linearize_index({i, j});
			ddb += node->grad[node_index];
			// what is the gradient for A? just add the node gradient back into it
			A->grad[node_index] += node->grad[node_index];

		}
		// add the gradient to b
		B->grad[j] += ddb;
	}

} 
