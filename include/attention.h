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
		

		layers.push_back(std::make_unique<LayerNorm>(std::vector<int>{2}, std::vector<int>{_embed_dim}));
		norm1 = dynamic_cast<LayerNorm*>(layers[layers.size()-1].get());

		layers.push_back(std::make_unique<LayerNorm>(std::vector<int>{2}, std::vector<int>{_embed_dim}));
		norm2 = dynamic_cast<LayerNorm*>(layers[layers.size()-1].get());
	}

	Tensor forward(Tensor t) override;

};

class Transformer : public Layer {
public:
	// basically hold a bunch of self attention blocks and add them...
	// we also want to take in an input dimension....
	// which we scale up with an MLP layer I guess
	
	// so it goes input embed (MLP), pos_embed -> N*transformer -> output embed (MLP) -> output

	MLP* input_embed;
	std::vector<SelfAttnBlock*> attn_blocks;
	MLP* output_embed;
	
	int embed_dim;
	
	// we want to have num_blocks, mlp ratio, input_embed_size
	// sequence length and path dont matter
	Transformer(
			int _num_blocks, 
			int _num_heads, 
			int _embed_dim, int _input_embed_dim, int _output_embed_dim, 
			int _mlp_ratio, 
			std::function<Tensor(const Tensor&)> _activation_fn) : embed_dim(_embed_dim) {
		
		// first crate the MLP
		layers.push_back(std::make_unique<MLP>(_input_embed_dim, _embed_dim, 2, _mlp_ratio, _activation_fn));
		input_embed = dynamic_cast<MLP*>(layers[layers.size()-1].get());
		
		// now the attention blocks
		for (int i=0; i<_num_blocks; i++) {
			layers.push_back(std::make_unique<SelfAttnBlock>(_num_heads, _embed_dim, _mlp_ratio, _activation_fn));
			attn_blocks.push_back(dynamic_cast<SelfAttnBlock*>(layers[layers.size()-1].get()));
		}

		// now the output
		layers.push_back(std::make_unique<MLP>(_embed_dim, _output_embed_dim, 2, _mlp_ratio, _activation_fn));
		output_embed = dynamic_cast<MLP*>(layers[layers.size()-1].get());
	
	}
	
	// uhhhhhh ok so its pretty easy
	Tensor forward(Tensor t) {
		
		// do the input embedding, this is
		// just a projection of the last dimension
		Tensor x = input_embed->forward(t);
		x = fixed_sincos_pos_embed(x, embed_dim);

		for (const auto& block : attn_blocks) {
			x = block->forward(x);
		}

		x = output_embed->forward(x);
		return x;

	}

};
