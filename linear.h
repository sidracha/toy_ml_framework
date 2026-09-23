#pragma once

#include <vector>
#include <random>
#include <stdexcept>

#include "tensor.h"

std::vector<Tensor*>* mm_plus_b_2d(std::vector<Tensor*>* input, std::vector<std::vector<Tensor*>>* weights, std::vector<Tensor*>* biases);

class Linear {
public:
	std::vector<std::vector<Tensor*>>* weights;
	std::vector<Tensor*>* biases;
	double DIST_MIN = 0.0;
	double DIST_MAX = 1.0;
	int N;
	int M;

	Linear(int _N, int _M) : N(_N), M(_M) {};

	void initialize_weights() {
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<double> dist(DIST_MIN, DIST_MAX);
		
		if (N <= 0 || M <= 0) throw std::runtime_error("Invalid linear layer shape");
		
		// initialize weights here
		double x;
		for (int i=0; i<N; i++) {
			for (int j=0; j<M; j++) {
				x = dist(gen);
				(*weights)[i][j] = new Tensor(x);
			}
		}
		
		// initialize biases here
		for (int i=0; i<M; i++) {
			x = dist(gen);
			(*biases)[i] = new Tensor(x);
		}

	}
	// we need to hold all of the weights
	// we also need to do some sort of randomized initialization? can do between 0-1 i guess since the biases will also be between 0-1
	
	// we want to return the matrix 2mm after the forward I guess
	std::vector<Tensor*>* forward(std::vector<Tensor*>* input) {
		return mm_plus_b_2d(input, weights, biases); 
	}

};
