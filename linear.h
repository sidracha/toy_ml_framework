#pragma once

#include <vector>
#include <random>
#include <stdexcept>

#include "tensor.h"



class Linear {
public:

	int N;
	int M;
	Graph graph;
	Tensor weight;
	Tensor bias;


	double DIST_MIN = 0.0;
	double DIST_MAX = 1.0;
	Linear(int _N, int _M)
		: N(_N),
		M(_M),
		graph(),
		weight(create_tensor_random({N, M}, &graph, DIST_MIN, DIST_MAX)),
		bias(create_tensor_random({M}, &graph, DIST_MIN, DIST_MAX)) {

		if (N <= 0 || M <= 0) throw std::runtime_error("Invalid linear layer shape");
	
	}

};
