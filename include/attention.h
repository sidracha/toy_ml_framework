#pragma once

#include "tensor.h"
#include "layer.h"
#include "linear.h"
#include "nn.h"
#include "modules.h"

class SelfAttn : public Layer {
public:
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

class SelfAttnBlock : public Layer {
public:
	
	SelfAttn* attn;
	MLP* mlp;

	LayerNorm* norm1;
	LayerNorm* norm2;

	SelfAttnBlock(int _num_heads, int _embed_dim, int _mlp_ratio, std::function<Tensor(const Tensor&)> _activation_fn) {
		layers.push_back(std::make_unique<SelfAttn>(_num_heads, _embed_dim));
		layers.push_back(std::make_unique<MLP>(_embed_dim, _embed_dim, 2, _mlp_ratio, _activation_fn));
		attn = dynamic_cast<SelfAttn*>(layers[0].get());
		mlp = dynamic_cast<MLP*>(layers[1].get());
		

		layers.push_back(std::make_unique<LayerNorm>(std::vector<int>{2}));
		norm1 = dynamic_cast<LayerNorm*>(layers[layers.size()-1].get());

		layers.push_back(std::make_unique<LayerNorm>(std::vector<int>{2}));
		norm2 = dynamic_cast<LayerNorm*>(layers[layers.size()-1].get());
	}

	Tensor forward(Tensor t) override;
	void zero_grad() override {
		for (auto& layer : layers) layer->zero_grad();
	}
	void gradient_descent_step(double lr) override {
		for (auto& layer : layers) layer->gradient_descent_step(lr);
	}
};
