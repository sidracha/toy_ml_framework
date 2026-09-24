#include "nn.h"
#include "tensor.h"

// ReLU is an operation and should return a Tensor, we also have to calculate the backward pass..
// for now its automatic
Tensor ReLU(Tensor t) {
	
	// create another tensor with the graph of the current tensor
	// we have to make sure the shape and the stride are the same... but when we create it with the shape,
	// it might be discontinuous...
	// do we create the gradient...?
	
	// i guess we can set the stride the same i guess...
	// create a tensor with the same shape and stride...
	
	Tensor new_tensor = create_tensor_clone_scalar(t, t.graph, 0.0);
	// now its a clone, do the ReLU
	for (int i=0; i<t.tensor_node->data.size(); i++) {
		int val;
		if (t.tensor_node->data[i] > 0.0) new_tensor.tensor_node->data[i] = 1.0; 
	}

	return t * new_tensor;

}

Tensor sigmoid(Tensor t) {

}
