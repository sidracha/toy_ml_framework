#pragma once

#include "tensor.h"
#include "layer.h"
#include "linear.h"
#include "nn.h"

class SelfAttn : public Layer {
public:
	std::vector<std::unique_ptr<Layer>> layers;
	int embed_dim;
	int hidden_dim;

	Linear* wq;
	Linear* wk;
	Linear* wv;

	SelfAttn(int _embed_dim, int _hidden_dim);
	Tensor forward(Tensor t) override;
	void zero_grad() override;
	void gradient_descent_step(double lr) override;

};
