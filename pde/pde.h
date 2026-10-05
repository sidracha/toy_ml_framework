#pragma once

#include <vector>

#include "tensor.h"
#include "dataset.h"

class AutoregDataset : public Dataset {
public:
	// just return a vector for each one
	std::vector<std::vector<Tensor>> epoch_vectors;
	std::vector<std::vector<Tensor>> tensor_vectors;

	AutoregDataset() {}

	// ok so tensor vectors will hold all the trajectories
	// we also need epoch vectors, will will concat them
	void shuffle();

	// shuffle the tensor vectors..
	// we have the tensors in the data tensors .
	// read them into the epoch vectors
	void prepare_epoch(int batch_size, int num_steps);
	std::vector<Tensor> get_input();
	
	// num_samples per epoch
	int num_samples() {return epoch_vectors.size();}

};

std::vector<double> fourier_initial_condition(int x_size, int k, double L);


Tensor delx_1d_periodic(Tensor pred, double delta_x, int op_dim);
void delx_1d_periodic_backward(TensorNode* ux);

Tensor del2x_1d_periodic(Tensor pred, double delta_x, int op_dim);
void del2x_1d_periodic_backward(TensorNode* uxx);
