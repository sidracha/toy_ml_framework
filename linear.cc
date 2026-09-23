#include "linear.h"

#include <vector>
#include <stdexcept>

std::vector<Tensor*>* mm_plus_b_2d(std::vector<Tensor*>* input, std::vector<std::vector<Tensor*>>* weights, std::vector<Tensor*>* biases) {
	int input_N = 1;
	int input_M = input->size();

	int weights_N = weights->size();
	int weights_M = (*weights)[0].size();
	
	if (input_M != weights_N) throw std::runtime_error("Shape mismatch");
	
	// resulting is just a vector of weights_M
	if (biases->size() != weights_M) throw std::runtime_error("Shape mismatch");
	
	// calculate the dot product here
	std::vector<Tensor*>* return_tensor = new std::vector<Tensor*>(weights_M);

	for (int i=0; i<weights_M; i++) {
		Tensor* val = *(*input)[0] * (*weights)[0][i];
		for (int j=1; j<input_M; j++) {
			val = *val + *(*input)[j] * (*weights)[j][i];
		}
		(*return_tensor)[i] = val;
	}
	
	// now add the biases
	for (int i=0; i<weights_M; i++) {
		(*return_tensor)[i] = *(*return_tensor)[i] + (*biases)[i];
	}
	return return_tensor;

} 
