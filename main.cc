#include <iostream>

#include "tensor.h"
#include "linear.h"

int main () {
	
	Graph g;

	Tensor input_tensor = create_tensor_zeros({10, 10}, &g);
	
	Linear linear1(10, 10);

	linear1.forward(input_tensor);

	return 0;
}
