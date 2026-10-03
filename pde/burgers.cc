// this file is all the stuff related to burgers
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

// burgers is in the form
// du/dt + u*du/dx = v*d^2u/dx^2
// du/dt = -u*du/dx + v*d^2u/dx^2
// and the forward euler step is just 
// u_t+1 = u_t + delta_t * (du/dt)

// du/dx = (u[i+1] - u[i-1])/(2*delta_x)

// d^u/dx^2 = (u[i+1] - 2*u[i] - u[i-1]) / (4*delta_x)


std::vector<double> burgers_forward_euler_step_periodic(std::vector<double>& u, double delta_x, double delta_t, double nu) {
	// do 1 forward euler step here
	// periodic boundary condition just means that u(0) = u(L), right? 
	// soo we can calcualte the update like that
		
	// for each point, calculate dudx;
	int N = u.size();
	std::vector<double> u_next(N);

	for (int i=0; i<u.size(); i++) {
		double ui_m1 = (i == 0) ? u[N-1] : u[i-1];
		double ui_p1 = (i == N-1) ? u[0] : u[i+1];

		double du_dx = (ui_p1 - ui_m1) / (2 * delta_x);
		double d2u_dx2 = (ui_p1 - 2*u[i] + ui_m1) / (delta_x * delta_x);
		
		// du/dt = -u*du/dx + v*d^2u/dx^2

		double du_dt = -u[i] * du_dx + nu * d2u_dx2;

		// now do the timestep for delta_t
		// u_t+1 = u_t + delta_t * (du/dt)
		u_next[i] = u[i] + delta_t * (du_dt);
		
	}
	return u_next;

}

void burgers_init_dataset(Dataset& train_dataset, Dataset& test_dataset, double L, int x_size, double delta_t, double nu) {
	
	int k = 3;
	double delta_x = L / (double) x_size;

	// we can create tensors like this, I guess...
	// 100 samples... we can use different initial conitions  and modes but thats it
	int NUM_SAMPLES = 100;
	// each data tensor is shape 1, N, 1
	for (int i=0; i<NUM_SAMPLES; i++) {
		std::vector<double> u = fourier_initial_condition(x_size, k, L);
				
		// timestep forward
		for (int j=0; j<400; j++) {

			std::vector<double> u_prev = u;
			Tensor input = create_tensor(u_prev, {1, x_size, 1});

			std::vector<double> u_next = burgers_forward_euler_step_periodic(u, delta_x, delta_t, nu);
			u = u_next;
			Tensor target = create_tensor(u_next, {1, x_size, 1});

			if (i < NUM_SAMPLES-1) train_dataset.data_tensors.push_back({input, target});
			else test_dataset.data_tensors.push_back({input, target});
		}
	}
}

// r = du/dt + u*du/dx - v * d^2u/dx^2
// we can calculate du/dx by just using our periodic 1d function
// and we can calculate d^2u/dx^2 by using the 2nd derivative function
Tensor burgers_residual(Tensor input, Tensor pred, double delta_x, double delta_t, double nu) {
	
	// we have the input and the pred... so we calculate all the spatial derivatives
	// directly from the pred
	
	// then we calcualte the du/dt as
	// (pred-input) / (deltat)
	Tensor ut = (pred - input) / (delta_t);
	
	// now calculate ux and uxx
	Tensor ux = delx_1d_periodic(pred, delta_x);
	Tensor uxx = del2x_1d_periodic(pred, delta_x);

	// now directly calculate the residual
	Tensor residual = ut + pred*ux - uxx * nu;
	// now... we want to do the reduction... but this is kinda cooked right?
	// since this is all vectors, lol

	return residual;

}
