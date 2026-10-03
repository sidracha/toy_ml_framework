// this file is all the stuff relative to burgers
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

std::vector<double> burgers_fourier_initial_condition(int x_size, int k, double L) {
	
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

void burgers_train_loop() {
	
	int x_size = 32;
	int k = 3;
	double L = 2*std::numbers::pi;
	double delta_x = L / x_size;
	double nu = 0.1;
	double delta_t = 0.03;

	// we can create tensors like this, I guess...
	// 100 samples... we can use different initial conitions  and modes but thats it
	Dataset train_dataset;
	Dataset test_dataset;
	int NUM_SAMPLES = 40;
	// each data tensor is shape 1, N, 1
	for (int i=0; i<NUM_SAMPLES; i++) {
		std::vector<double> u = burgers_fourier_initial_condition(x_size, k, L);
				
		// timestep forward
		for (int j=0; j<150; j++) {

			std::vector<double> u_prev = u;
			Tensor input = create_tensor(u_prev, {1, x_size, 1});

			std::vector<double> u_next = burgers_forward_euler_step_periodic(u, delta_x, delta_t, nu);
			u = u_next;
			Tensor target = create_tensor(u_next, {1, x_size, 1});

			if (i < NUM_SAMPLES-1) train_dataset.data_tensors.push_back({input, target});
			else test_dataset.data_tensors.push_back({input, target});
		}
	}


	int N = x_size;	
	int num_blocks = 4;
	int num_heads = 4;
	int embed_dim = 8;
	int input_embed_dim = 1;
	int output_embed_dim = 1;
	int mlp_ratio = 2;
	auto identity = [](const Tensor& t) { return t; };
	Transformer model(num_blocks, num_heads, embed_dim, input_embed_dim, output_embed_dim, mlp_ratio, Tanh);
	
	double lr = 0.08;
	double lr_multiplier = 0.92; 
	Optimizer optimizer(model.layers, lr);

	// now do the training loop
	// can we even tell how many epochs there are? 
	int batch_size = 2;
	int epoch_size = train_dataset.data_tensors.size() / batch_size;
	int num_epochs = 16;
	
	// train loop
	for (int epoch=0; epoch<num_epochs; epoch++) {
		train_dataset.prepare_epoch(batch_size);
		for (int i=0; i<epoch_size; i++) {
			auto [input, target] = train_dataset.get_input();

			Tensor pred = model.forward(input);

			Tensor loss = MSELoss(pred, target);
			std::cout << "epoch: " << epoch << " | iteration: " << i << " loss: " << loss.data_at(0) << std::endl;
			
			optimizer.zero_grad();
			loss.backward();
			optimizer.step();

		}
		optimizer.lr *= lr_multiplier;
	}
	
	std::cout << std::endl;
	std::cout << "---------- VALIDATION ----------";
	std::cout << std::endl;
	std::vector<std::vector<double>> preds;
	std::vector<std::vector<double>> targets;
	
	int test_size = test_dataset.data_tensors.size();
	//std::optional<Tensor> input = std::nullopt;
	for (int i=0; i<test_size; i++) {
		auto& [input, target] = test_dataset.data_tensors[i];
		//if (!input.has_value()) input = _; 

		Tensor pred = model.forward(input);
		Tensor loss = MSELoss(pred, target);
		std::cout << "iteration: " << i << " loss: " << loss.data_at(0) << std::endl;
	
		preds.push_back(pred.tensor_node->data);
		targets.push_back(target.tensor_node->data);
		//input = pred;
	}

  save_burgers_video(preds, targets, "burgers_comparison");

}
