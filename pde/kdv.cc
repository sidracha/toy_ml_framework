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

// ut + uux + delta^2*uxxx = 0
// ut = -u*ux - delta^2*uxxx
std::vector<double> kdv_forward_euler_step_periodic(std::vector<double>& u, double delta_x, double delta_t, double delta) {
	
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
	
		double uxxx = (uxx_vector[ip] - uxx_vector[im]) / (2 * delta_x);
		double ux = (u[ip] - u[im]) / (2 * delta_x);
		double dudt = -u[i] * ux - (delta*delta) * uxxx; 

		// now apply the euler update
		u_next[i] = u[i] + delta_t * dudt;
	}
	return u_next;

}


void kdv_init_dataset(Dataset& train_dataset, Dataset& test_dataset) {
	
	int x_size = 64;
	int k = 3;
	double L = 2*std::numbers::pi;
	double delta_x = L / x_size;
	double delta_t = 0.005;
	double delta = 0.022;

	// we can create tensors like this, I guess...
	// 100 samples... we can use different initial conitions  and modes but thats it
	int NUM_SAMPLES = 80;
	// each data tensor is shape 1, N, 1
	for (int i=0; i<NUM_SAMPLES; i++) {
		std::vector<double> u = fourier_initial_condition(x_size, k, L);
				
		// timestep forward
		for (int j=0; j<1000; j++) {

			std::vector<double> u_prev = u;
			Tensor input = create_tensor(u_prev, {1, x_size, 1});

			std::vector<double> u_next = kdv_forward_euler_step_periodic(u, delta_x, delta_t, delta);
			u = u_next;
			Tensor target = create_tensor(u_next, {1, x_size, 1});

			if (i < NUM_SAMPLES-1) train_dataset.data_tensors.push_back({input, target});
			else test_dataset.data_tensors.push_back({input, target});
		}
	}
}
