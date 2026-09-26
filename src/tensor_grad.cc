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


// THE BACKWARD PASS FOR EACH OPERATOR ON THE TENSOR
// ASSUMES THAT GRAD FOR THE TENSOR IS CURRENTLY ALREADY CALCULATED
// BACKPROP THE GRADIENTS IN THE CURRENT ONE TO THE CHILDREN


// For now we assume shape == stride == same for all of these pointwise
// operations, should all work with 3D tensors automatically

bool verify_predecessor_size(TensorNode* node, int expected) {
	if (node->predecessors.size() != expected) throw std::runtime_error("Predecessor size is not what is expected"); 
}


// ADD HAS TO BE OF THE SAME SHAPE
void add_backward(TensorNode* node) {
	verify_predecessor_size(node, 2);

	TensorNode* a = node->predecessors[0].get();
	TensorNode* b = node->predecessors[1].get();

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

	TensorNode* a = node->predecessors[0].get();
	TensorNode* b = node->predecessors[1].get();

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

	TensorNode* a = node->predecessors[0].get();
	TensorNode* b = node->predecessors[1].get();

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

	TensorNode* a = node->predecessors[0].get();
	TensorNode* b = node->predecessors[1].get();

	for (int i=0; i<node->grad.size(); i++) {
		double dda = 1/b->data[i];
		double b_data = b->data[i];
		double ddb = -a->data[i] * (1/(b_data * b_data));
		a->grad[i] += (node->grad[i] * dda);
		b->grad[i] += (node->grad[i] * ddb);

	}

}

// lets say C = AB for this
// ddA = dC * B^T
// ddB = A^T * dC

void MATMUL_2D_backward(TensorNode* node) {
	verify_predecessor_size(node, 2);

	TensorNode* A = node->predecessors[0].get();
	TensorNode* B = node->predecessors[1].get();
	// ddA = dC * B^T
	B->transpose();
	BATCHED_GEMM_2D_ADD (
		node->grad,
		node->stride,
		node->shape,
		B->data,
		B->stride,
		B->shape,
		A->grad,
		A->stride,
		A->shape
	);
	B->transpose();
	
	// ddB kinda annoying cuz its only 2D....
	// so it contributes to all of it in node
	// so we have to do like the odometer stuff its 
	// super annoying i hate doing this odometer shit
	A->transpose();

	int dim = A->shape.size();
	int index_N = dim - 2;
	int index_M = dim - 1;

	int M = A->shape[index_N];
	int N = A->shape[index_M];
	int K = node->shape[index_M];

	std::vector<int> odometer_A;
	std::vector<int> odometer_node;
	for (int i = 0; i < index_N; i++) {
		odometer_A.push_back(0);
		odometer_node.push_back(0);
	}

	// so in this case A index and B index are fine
	// but ugh has to be extendable
	// but technically......
	// we do C = AB
	// ... and C is Node
	// but then C can have different dimensions to A...
	// but then how will just moving that odometer index forward
	// oh wait yeah its cuz of the first sahpes are the same
	// but still better cuz it might have differnet strides and stuff, just for the future i guess
	int odometer_index_A = 0;
	int odometer_index_node = 0;

	while (odometer_index_A >= 0) {

		for (int i = 0; i < M; i++) {
			for (int j = 0; j < K; j++) {
				double dot_product = 0.0;
				for (int k = 0; k < N; k++) {
					int A_index = odometer_index_A + i * A->stride[index_N] + k * A->stride[index_M];
					int dC_index = odometer_index_node + k * node->stride[index_N] + j * node->stride[index_M];
					dot_product += A->data[A_index] * node->grad[dC_index];
				}
				int B_index = i * B->stride[0] + j * B->stride[1];
				B->grad[B_index] += dot_product;
			}
		}
		odometer_index_A = odometer_next(odometer_A, A->shape, A->stride);
		odometer_index_node = odometer_next(odometer_node, node->shape, node->stride);
	}

	A->transpose();
}


void BIAS_ADD_2D_1D_backward(TensorNode* node) {
	
	
	// soooo A is gonna be the full matrix

	TensorNode* A = node->predecessors[0].get();
	TensorNode* B = node->predecessors[1].get();
	int dim_A = A->dim();
	int index_N = dim_A-2;
	int index_M = dim_A-1;
	int N = A->shape[index_N];
	int M = A->shape[index_M];

	// uhhhhh we prob need the odometer here, right? 
	// basically just add all the contributions since its column wise
	// and for a, the grad is just all 1s, so its fine we just add it 
	
	// okkk so we loop over all of the batch as well... and we sum the gradients from the entire batch
	// the entire batch also gets summed into ddb 
	// but for ddA, we just want to add the gradient from the particular one also into the batch one
		
	std::vector<int> odometer_node;
	std::vector<int> odometer_A;
	for (int i=0; i<index_N; i++) {
		odometer_node.push_back(0);
		odometer_A.push_back(0);
	}
	int odometer_index_node = 0;
	int odometer_index_A = 0;


	for (int j=0; j<M; j++) {

		double ddb = 0.0;
		for (int i=0; i<odometer_node.size(); i++) {
			odometer_node[i] = 0;
			odometer_A[i] = 0;
		}
		odometer_index_node = 0;
		odometer_index_A = 0;

		while (odometer_index_node >= 0) {
			for (int i=0; i<N; i++) {

				// uhhh we have to calculate the index for the node and the bias
				// or whatever
				int node_index = odometer_index_node + i * node->stride[index_N] + j * node->stride[index_M];
				int A_index = odometer_index_A + i * A->stride[index_N] + j * A->stride[index_M];

				ddb += node->grad[node_index];
				A->grad[A_index] += node->grad[node_index];

			}
			odometer_index_node = odometer_next(odometer_node, node->shape, node->stride);
			odometer_index_A = odometer_next(odometer_A, A->shape, A->stride);

		}
		int B_index = j * B->stride[0];
		B->grad[B_index] += ddb;
	}
}

