#pragma once

#include <vector>
#include <random>
#include <stdexcept>

#include "tensor.h"
#include "layer.h"
#include "nn.h"

class Linear : public Layer {
public:

	double DIST_MIN = -0.5;
	double DIST_MAX = 0.5;
	int N;
	int M;
	Tensor weight;
	Tensor bias;


	Linear(int _N, int _M)
		: N(_N),
		M(_M),
		weight(create_tensor_random({N, M}, DIST_MIN, DIST_MAX)),
		bias(create_tensor_zeros({M})) {

		if (N <= 0 || M <= 0) throw std::runtime_error("Invalid linear layer shape");
	
	}
	
	// tensor T is not owned by anything else...
	// each layers owns its own tensors, its fine!
	Tensor forward(Tensor t) override;	
	void zero_grad() override;
	void gradient_descent_step(double lr) override;

};

class LinearReLU : public Linear {
public:
	LinearReLU (int _N, int _M) : Linear(_N, _M) {}	
	Tensor forward(Tensor t) override;
};


class LinearSigmoid : public Linear {
public:
	LinearSigmoid (int _N, int _M) : Linear(_N, _M) {}	
	Tensor forward(Tensor t) override;
};


class MLP : public Layer {
public:
	int input_dim;
	int output_dim;
	int depth;
	int mlp_ratio;
	std::function<Tensor(const Tensor&)> activation_fn;

	MLP(int _input_dim, int _output_dim, int _depth, int _mlp_ratio, std::function<Tensor(const Tensor&)> _activation_fn) : 
		Layer(),
		input_dim(_input_dim),
		output_dim(_output_dim),
		depth(_depth),
		mlp_ratio(_mlp_ratio),
		activation_fn(_activation_fn){
		

		for (int i=0; i<depth; i++) {
		
			if (i == 0) layers.push_back(std::make_unique<Linear>(input_dim, mlp_ratio*input_dim));
			else if (i == depth-1) layers.push_back(std::make_unique<Linear>(mlp_ratio*input_dim, output_dim));
			else layers.push_back(std::make_unique<Linear>(mlp_ratio*input_dim, mlp_ratio*input_dim));
		}

	}

	Tensor forward(Tensor t) {
		
		// add the activation in the middle
		// but not at the end
		for (int i=0; i<depth; i++) {
			t = layers[i]->forward(t);
			
			if (i < depth-1) t = activation_fn(t); 
		}
		return t;

	}

	void zero_grad() {
		for (const auto& layer : layers) layer->zero_grad();
	}

	void gradient_descent_step(double lr) {
		for (const auto& layer : layers) layer->gradient_descent_step(lr);
	} 
	
	std::vector<std::unique_ptr<Layer>>& get_layers() {
		return layers;
	}

};
