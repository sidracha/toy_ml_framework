#include "pde.h"

#include <vector>
#include <numbers>
#include <random>
#include <cmath>
#include <iostream>


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
Tensor delx_1d_periodic(Tensor t, double delta_x) {
	
	// for each value of the tensor, calculate it using the finite differnece
	int N = t.tensor_node->data.size();
	std::vector<double> ux(N);
	for (int i=0; i<t.tensor_node->data.size(); i++) {
		int i1 = (i == 0) ? N-1 : i-1;
		int i2 = (i == N-1) ? 0 : i+1;

		ux[i] = (t.tensor_node->data[i2] - t.tensor_node->data[i1]) / (2 * delta_x);
	}
	
	std::shared_ptr<TensorNode> node = make_operator_output_node(
		ux, t.shape(), {t.tensor_node}, delx_1d_periodic_backward);
	
	node->delta_x = delta_x;
	return Tensor(node);
}

// ok so we need to do the backward here
// soo
// duxi/dui+1 = 1/(2*delta_x)
// duxi/dui-1 = -1/(2*delta_x)
void delx_1d_periodic_backward(TensorNode* ux) {
	
	TensorNode* u = ux->predecessors[0].get();
	
	int N = u->grad.size();
	for (int i=0; i<N; i++) {
		int i1 = (i == 0) ? N-1 : i-1;
		int i2 = (i == N-1) ? 0 : i+1;
			
		double delta_x = ux->delta_x;
		double duxi_duip1 = 1 / (2 * delta_x);
		double duxi_duim1 = -1 / (2 * delta_x);
	
		// use the chain rule
		// df/dui+1 = duxi/dui+1 * df/duxi
		u->grad[i1] += (duxi_duim1 * ux->grad[i]);
		u->grad[i2] += (duxi_duip1 * ux->grad[i]);

	}

}

// calcualte the 2nd derivative here directly
// uxx[i] = (u[i+1] - 2u[i] + u[i-1]) / (deltax^2)
Tensor del2x_1d_periodic(Tensor t, double delta_x) {

	int N = t.tensor_node->data.size();
	std::vector<double> uxx(N);

	for (int i=0; i<t.tensor_node->data.size(); i++) {
		int i1 = (i == 0) ? N-1 : i-1;
		int i2 = (i == N-1) ? 0 : i+1;
		
		uxx[i] = (t.tensor_node->data[i2] - 
							2*t.tensor_node->data[i] + 
							t.tensor_node->data[i1]) / (delta_x*delta_x);

	}
	

	std::shared_ptr<TensorNode> node = make_operator_output_node(
		uxx, t.shape(), {t.tensor_node}, del2x_1d_periodic_backward);
	
	node->delta_x = delta_x;
	return Tensor(node);

}

// backward on the 2nd derivative directly
// uxx[i] = (u[i+1] - 2u[i] + u[i-1]) / (deltax^2)
//ddi+1 = 1/(dx^2)
//ddi = -2/(dx^2)
//ddi-2 = 1/(dx^2)
void del2x_1d_periodic_backward(TensorNode* uxx) {

	TensorNode* u = uxx->predecessors[0].get();
	
	int N = u->grad.size();
	for (int i=0; i<N; i++) {
		int im1 = (i == 0) ? N-1 : i-1;
		int ip1 = (i == N-1) ? 0 : i+1;
	
		double delta_x = uxx->delta_x;	
		double dx2 = delta_x * delta_x;
		
		double ddip1 = 1 / (dx2);
		double ddi = -2 / (dx2);
		double ddim1 = 1 / (dx2);
		
		u->grad[im1] += (ddim1 * uxx->grad[i]);
		u->grad[i] += (ddi * uxx->grad[i]);
		u->grad[ip1] += (ddip1 * uxx->grad[i]);

	}
}
