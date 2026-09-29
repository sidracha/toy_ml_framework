#pragma once

#include "tensor.h"
#include "layer.h"
#include "linear.h"
#include "nn.h"

class SelfAttn : public Layer {
public:
	std::vector<std::unique_ptr<Layer>> layers;
	int num_heads;
	int embed_dim;
	int head_dim;

	std::vector<Linear*> wq;
	std::vector<Linear*> wk;
	std::vector<Linear*> wv;
	
	Linear* output_projection_layer;

	SelfAttn(int _num_heads, int _embed_dim);
	Tensor forward(Tensor t) override;
	void zero_grad() override;
	void gradient_descent_step(double lr) override;

};
