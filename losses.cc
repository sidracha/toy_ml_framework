#include "tensor.h"
#include "losses.h"

#include <cmath>
#include <stdexcept>

// we want to create the loss output node into the graph of the current, not into the graph of the target,
// which is onwed by something else

// eh fuck it only support 2D or 1D MSELoss right now
Tensor MSELoss(Tensor a, Tensor b) {
	if (a.dim() != 2 || b.dim() != 2) throw std::runtime_error("Sorry, only dim=2 is supported for MSELoss at this time");
	if (a.stride() != b.stride() || a.shape() != b.shape()) throw std::runtime_error("Shape mismatch"); 

	// each row holds a "vector"
	// so I guess calculate the MSE of the vectors first
	
	// it should create a 1-D tensor at the end.. and we need to attach a backwards fucntion
	// how can we do this?
	
	// this is JUST the forward pass I guess
	std::vector<double> output(1);
	
	int N = a.shape()[0];
	int M = a.shape()[1];
	for (int i=0; i<N; i++) {
		double row_sum = 0.0;
		for (int j=0; j<M; j++) {
			int a_index = a.linearize_index({i, j});
			int b_index = b.linearize_index({i, j});
			row_sum += (std::pow((a.tensor_node->data[a_index] - b.tensor_node->data[b_index]), 2)) / static_cast<double>(M);
		}
		output[0] += row_sum / static_cast<double>(N);
	} 
	
	//we have to make the node, I guess? 
	// put it in tensor As graph. should be like (pred, target)
	TensorNode* raw = make_operator_output_node(output, {1}, Op::OTHER, a.tensor_node, b.tensor_node, a.graph, MSELoss_backward);
	return Tensor(raw, a.graph);

}


// how to calcualte the backward pass...???
// this is pretty easy, actually.
// lets fuse the nodes too, I guess

// dda = (2/mn) * (a-b)
// ddb = -(2/mn) * (a-b)
void MSELoss_backward(TensorNode* node) {
	
	TensorNode* a = node->predecessors[0];
	TensorNode* b = node->predecessors[1];
	
	// just write the values in, tihs is pretty easy
	int N = a->shape[0];
	int M = a->shape[1];
	// multiply also by the grad of the current node

	for (int i=0; i<N; i++) {
		for (int j=0; j<M; j++) {
			int a_index = a->linearize_index({i, j});
			int b_index = b->linearize_index({i, j});
			
			double first_term = 2 / M*N;
			double second_term = (a->data[a_index] - b->data[b_index]);
		
			double dda = node->grad[0] * first_term * second_term;
			double ddb = -1.0 * node->grad[0] * first_term * second_term;

			a->grad[a_index] += dda;
			b->grad[b_index] += ddb;
 
		}
	}

}
