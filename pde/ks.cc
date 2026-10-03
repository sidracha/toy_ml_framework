#include <vector>
#include <cmath>
#include <utility>
#include <numbers>
#include <random>
#include <iostream>
#include <optional>

#include "burgers.h"
#include "attention.h"
#include "nn.h"
#include "optimizer.h"
#include "losses.h"
#include "dataset.h"
#include "visualize.h"
#include "pde.h"

// make the euler step here....
// ut = -uxx - uxxxx - 1/2ux^2
//
// how to calculate uxx? we can do this same as in burgers
// but uxxxx we can just do the stencil grid again and again....
//
//
// so uxxxx = (uxxi+1 - 2uxxi + uxxi-1) / deltax^2
// euler update: u_t+1 = u_t + delta_t * du/dt 
std::vector<double> ks_forward_euler_step_periodic(std::vector<double>& u, double delta_x, double delta_t) {
	
	// for each one apply a single step
	int N = u.size();
	std::vector<double> u_next(N);
	std::vector<double> uxx_vector(N);
	
	// calculate uxx here
	for (int index=0; index<N; index++) {
		int i = index;
		int im = (index == 0) ? N-1 : index-1;
		int ip = (index == N-1) ? 0 : index+1;
		uxx_vector[i] = (u[ip] - 2*u[i] + u[im]) / (delta_x * delta_x);

	}

	// now run the full loop to do the update
	for (int index=0; index<N; index++) {
		int i = index;
		int im = (index == 0) ? N-1 : index-1;
		int ip = (index == N-1) ? 0 : index+1;

		double ux = (u[ip] - u[im]) / (2 * delta_x);
		double uxx = uxx_vector[i];
		double uxxxx = (uxx_vector[ip] - 2*uxx_vector[i] + uxx_vector[im]) / (delta_x * delta_x);
		double dudt = -uxx - uxxxx - u[i]*ux;

		// now apply the euler update
		u_next[i] = u[i] + delta_t * dudt;
	}
	return u_next;

}


void ks_init_dataset(Dataset& train_dataset, Dataset& test_dataset) {
	
	int x_size = 40;
	int k = 3;
	double L = 8*std::numbers::pi;
	double delta_x = L / x_size;
	double delta_t = 0.001;

	// we can create tensors like this, I guess...
	// 100 samples... we can use different initial conitions  and modes but thats it
	int NUM_SAMPLES = 80;
	// each data tensor is shape 1, N, 1
	for (int i=0; i<NUM_SAMPLES; i++) {
		std::vector<double> u = fourier_initial_condition(x_size, k, L);
				
		// timestep forward
		for (int j=0; j<100; j++) {

			std::vector<double> u_prev = u;
			Tensor input = create_tensor(u_prev, {1, x_size, 1});

			std::vector<double> u_next = ks_forward_euler_step_periodic(u, delta_x, delta_t);
			u = u_next;
			Tensor target = create_tensor(u_next, {1, x_size, 1});

			if (i < NUM_SAMPLES-1) train_dataset.data_tensors.push_back({input, target});
			else test_dataset.data_tensors.push_back({input, target});
		}
	}
}
