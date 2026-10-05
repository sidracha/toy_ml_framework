#include "pde.h"
#include "utils.h"
#include "nn.h"

#include <vector>
#include <numbers>
#include <random>
#include <cmath>
#include <iostream>
#include <algorithm>


std::vector<double> fourier_initial_condition(int x_size, int k, double L) {
	
	std::random_device rd; 
	std::mt19937 gen(rd()); 
	
	// this is the scaling factor for the coefficients
	// will be -A/k^2, A/k^2
	double A = 1.0;

	// make the combinatinations of sines and cosines here
	
	// vector will hold k, a_k, b_k
	std::vector<std::vector<double>> modes;

	// we can let k go from 1-k nfor now and change it later
	for (int i=1; i<=k; i++) {
		
		double dist_min = -A / (i*i);
		double dist_max = A / (i*i);
		std::uniform_real_distribution<double> dist(dist_min, dist_max);
		double a_k = dist(gen);
		double b_k = dist(gen);

		modes.push_back({(double)i, a_k, b_k});
	}

	// now create the initial function...
	// we can just seed it for all x
	
	std::vector<double> domain(x_size);
	// x_size is the number of points. so each point x will be from 
	// i=0....x_size-1 x = i * L/x_size
	
	double spacing = L / (double) x_size;
	for (int i=0; i<x_size; i++) {
		double x = i * spacing;
		
		// now for the current x, generate the combination from the 
		// fourier initial condition
		// will of form a_k * sin(2pikx/L) + b_k * cos(2pikx/L)
		double value = 0;
		for (std::vector<double> mode : modes) {
			double k = mode[0];
			double a_k = mode[1];
			double b_k = mode[2];
			double inside = 2 * std::numbers::pi * k * x / L;
			double sin_term = std::sin(inside);
			double cos_term = std::cos(inside);
			value += a_k * sin_term + b_k * cos_term;
		}

		domain[i] = value;
	}

	return domain;
		
}

// calculate here the first derivative of a Tensor
// this is just calculated as ux[i] = (u[i+1] - u[i-1]) / deltax
Tensor delx_1d_periodic(Tensor t, double delta_x, int op_dim) {
	
	// for each value of the tensor, calculate it using the finite differnece
	int N = t.tensor_node->data.size();
	std::vector<double> ux(N);
	
	// 1d operation, operate over one dimension
	// odometer with the rest
	std::vector<int> odometer;
	std::vector<int> index_map;
	int dim = t.dim();
	for (int i=0; i<dim; i++) {
		if (i != op_dim) {
			odometer.push_back(0);
			index_map.push_back(i);
		}
	}
	std::vector<int> shape = t.shape();
	std::vector<int> stride = t.stride();

	bool has_next = true;
	while (has_next) {
		int n = shape[op_dim];
		for (int i=0; i<n; i++) {
			int odometer_offset = calculate_offset(odometer, stride, index_map);
			int im = (i == 0) ? n-1 : i-1;
			int ip = (i == n-1) ? 0: i+1;
			int index = odometer_offset + i*stride[op_dim];
			int im_index = odometer_offset + im*stride[op_dim];
			int ip_index = odometer_offset + ip*stride[op_dim];
			
			ux[index] = (t.tensor_node->data[ip_index] - t.tensor_node->data[im_index]) / (2*delta_x);
			
		}
		has_next = odometer_next_mapped(odometer, shape, index_map);
	}
	
	std::shared_ptr<TensorNode> node = make_operator_output_node(
		ux, t.shape(), {t.tensor_node}, delx_1d_periodic_backward);
	
	node->delta_x = delta_x;
	node->del_op_dim = op_dim;
	return Tensor(node);
}

// ok so we need to do the backward here
// soo
// duxi/dui+1 = 1/(2*delta_x)
// duxi/dui-1 = -1/(2*delta_x)
void delx_1d_periodic_backward(TensorNode* ux) {
	
	TensorNode* u = ux->predecessors[0].get();

	int op_dim = ux->del_op_dim;
	std::vector<int> odometer;
	std::vector<int> index_map;
	
	int dim = ux->dim();
	for (int i=0; i<dim; i++) {
		if (i != op_dim) {
			odometer.push_back(0);
			index_map.push_back(i);
		}
	}

	std::vector<int> shape = ux->shape;
	std::vector<int> stride = ux->stride;
	
	double delta_x = ux->delta_x;
	double duxi_duip1 = 1 / (2 * delta_x);
	double duxi_duim1 = -1 / (2 * delta_x);
	
	bool has_next = true;
	while (has_next) {
		int n = ux->shape[op_dim];
		for (int i=0; i<n; i++) {
			
			int odometer_offset = calculate_offset(odometer, stride, index_map);
			int im = (i == 0) ? n-1 : i-1;
			int ip = (i == n-1) ? 0: i+1;
			int index = odometer_offset + i*stride[op_dim];
			int im_index = odometer_offset + im*stride[op_dim];
			int ip_index = odometer_offset + ip*stride[op_dim];

		
			// use the chain rule
			// df/dui+1 = duxi/dui+1 * df/duxi
			u->grad[im_index] += (duxi_duim1 * ux->grad[index]);
			u->grad[ip_index] += (duxi_duip1 * ux->grad[index]);
		}
		has_next = odometer_next_mapped(odometer, shape, index_map);

	}

}

// calcualte the 2nd derivative here directly
// uxx[i] = (u[i+1] - 2u[i] + u[i-1]) / (deltax^2)
Tensor del2x_1d_periodic(Tensor t, double delta_x, int op_dim) {
	
	int N = t.tensor_node->data.size();
	std::vector<double> uxx(N);
	
	std::vector<int> odometer;
	std::vector<int> index_map;
	int dim = t.dim();
	for (int i=0; i<dim; i++) {
		if (i != op_dim) {
			odometer.push_back(0);
			index_map.push_back(i);
		}
	}
	std::vector<int> shape = t.shape();
	std::vector<int> stride = t.stride();

	bool has_next = true;
	while (has_next) {
		int n = shape[op_dim];
		for (int i=0; i<n; i++) {
			int odometer_offset = calculate_offset(odometer, stride, index_map);
			int im = (i == 0) ? n-1 : i-1;
			int ip = (i == n-1) ? 0: i+1;
			int index = odometer_offset + i*stride[op_dim];
			int im_index = odometer_offset + im*stride[op_dim];
			int ip_index = odometer_offset + ip*stride[op_dim];

			uxx[index] = (t.tensor_node->data[ip_index] - 
							2*t.tensor_node->data[index] + 
							t.tensor_node->data[im_index]) / (delta_x*delta_x);
			
			
		}
		has_next = odometer_next_mapped(odometer, shape, index_map);
	}
	
	std::shared_ptr<TensorNode> node = make_operator_output_node(
		uxx, t.shape(), {t.tensor_node}, del2x_1d_periodic_backward);
	
	node->delta_x = delta_x;
	node->del_op_dim = op_dim;
	return Tensor(node);
}

// backward on the 2nd derivative directly
// uxx[i] = (u[i+1] - 2u[i] + u[i-1]) / (deltax^2)
//ddi+1 = 1/(dx^2)
//ddi = -2/(dx^2)
//ddi-2 = 1/(dx^2)
void del2x_1d_periodic_backward(TensorNode* uxx) {
	
	TensorNode* u = uxx->predecessors[0].get();

	int op_dim = uxx->del_op_dim;
	std::vector<int> odometer;
	std::vector<int> index_map;
	
	int dim = uxx->dim();
	for (int i=0; i<dim; i++) {
		if (i != op_dim) {
			odometer.push_back(0);
			index_map.push_back(i);
		}
	}

	std::vector<int> shape = uxx->shape;
	std::vector<int> stride = uxx->stride;

	double delta_x = uxx->delta_x;
	double dx2 = (delta_x * delta_x);

	double ddip = 1 / (dx2);
	double ddi = -2 / (dx2);
	double ddim = 1 / (dx2);
	
	bool has_next = true;
	while (has_next) {
		int n = uxx->shape[op_dim];
		for (int i=0; i<n; i++) {
			
			int odometer_offset = calculate_offset(odometer, stride, index_map);
			int im = (i == 0) ? n-1 : i-1;
			int ip = (i == n-1) ? 0: i+1;
			int index = odometer_offset + i*stride[op_dim];
			int im_index = odometer_offset + im*stride[op_dim];
			int ip_index = odometer_offset + ip*stride[op_dim];

		
			// use the chain rlue
			u->grad[index] += (ddi * uxx->grad[index]);
			u->grad[ip_index] += (ddip * uxx->grad[index]);
			u->grad[im_index] += (ddim * uxx->grad[index]);

		}
		has_next = odometer_next_mapped(odometer, shape, index_map);

	}

}

// ok so tensor vectors will hold all the trajectories
// we also need epoch vectors, will will concat them
void AutoregDataset::shuffle() {

	std::random_device rd;
	std::mt19937 gen(rd());

	std::shuffle(tensor_vectors.begin(), tensor_vectors.end(), gen);
}

// shuffle the tensor vectors..
// we have the tensors in the data tensors .
// read them into the epoch vectors
void AutoregDataset::prepare_epoch(int batch_size, int num_steps) {

	index = 0;
	epoch_vectors.clear();
	tensor_vectors.clear();

	int n = data_tensors.size();
	if (n % num_steps != 0) throw std::runtime_error("Prepare epoch: n != num_steps");
	int num_traj = n / num_steps;

	for (int i=0; i<num_traj; i++) {

		std::vector<Tensor> v;
		for (int j=0; j<num_steps; j++) {

			int index = i * num_steps + j;
			v.push_back(data_tensors[index].first);
		}
		tensor_vectors.push_back(v);
	}

	// now we have to do the batch size of them...
	// each one of tensor_vectors contains an independent trajecotry
	// so we have to make all of them one tensor per position

	int nt = tensor_vectors.size();
	if (nt % batch_size != 0) {
		throw std::runtime_error("tensor vectors size should be divisible by batch size");
	}
	int num_batches = nt / batch_size;

	// now we should shuffle tensor vectors? cuz we want our batches to be
	// randomized
	shuffle();
	// now go through for each batch size...
	for (int i=0; i<num_batches; i++) {

		std::vector<Tensor> batched_trajectory;
		for (int j=0; j<num_steps; j++) {
			// ok so for each one inside the batch size we want to iterate through
			// the position k of it and concatenate them together....
			// so this batch should be {tensor[0][0], tensor[1][0], tensor[2][0]} and concat them

			// this is for each position, we have to cat it
			std::vector<Tensor> positions;
			for (int k=0; k<batch_size; k++) {
				// this is the in dex of the VECTOR
				int index = i * batch_size + k;
				positions.push_back(tensor_vectors[index][j]);
			}
			Tensor batched_position = concat(positions, 0);
			// okk so this the batched tensor at some position... but this needs to be appended into the
			// vector of trajectory
			batched_trajectory.push_back(batched_position);
		}
		epoch_vectors.push_back(batched_trajectory);
	}

}

std::vector<Tensor> AutoregDataset::get_input() {
	return epoch_vectors[index++];
}
