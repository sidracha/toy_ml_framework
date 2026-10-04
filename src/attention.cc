#include "attention.h"

#include <iostream>

SelfAttn::SelfAttn(int _num_heads, int _embed_dim) {
	
	num_heads = _num_heads;
	embed_dim = _embed_dim;
	if (embed_dim % num_heads != 0) {
		throw std::runtime_error("embed_dim must be divisible by num_heads");
	}
	head_dim = embed_dim / num_heads;

	// ok... so we want an array for each of wq, wk, wv
	// but push the layers into the layers thing too cuz unique
	
	for (int i=0; i<num_heads; i++) {
		// make another unique linear whatever whatver
		layers.push_back(std::make_unique<Linear>(embed_dim, head_dim));
		layers.push_back(std::make_unique<Linear>(embed_dim, head_dim));
		layers.push_back(std::make_unique<Linear>(embed_dim, head_dim));

		// add the linear layers...
		// ohh hmmm we want to embed the embedding_dim
		// into hidden_dim

		// uhh ok so we want to embed each of htem then do the matmuls and the
		// transposes and shit
		
		int base = i * 3;
		wq.push_back(dynamic_cast<Linear*>(layers[base].get()));
		wk.push_back(dynamic_cast<Linear*>(layers[base+1].get()));
		wv.push_back(dynamic_cast<Linear*>(layers[base+2].get()));

	}
	
	// this is the output proj layer.... should we keep this in for single 
	// headed self attn? 
	// yeah keep it i guess
	layers.push_back(std::make_unique<Linear>(embed_dim, embed_dim));
	output_projection_layer = dynamic_cast<Linear*>(layers[layers.size()-1].get());
	
}

Tensor SelfAttn::forward(Tensor t) {
	
	
	// for each head do the qkv projection
	std::vector<Tensor> heads;
	
	for (int i=0; i<num_heads; i++) {
		Tensor proj_q = wq[i]->forward(t);
		// we have all the projection layers....
		// now take the output_tensors and put them all
		// in a vector do this for each head

		// transpose proj k
		Tensor proj_k = wk[i]->forward(t);
		Tensor proj_v = wv[i]->forward(t);

		
		proj_k = transpose(proj_k);
		double scale = std::pow(head_dim, 0.5);
		Tensor scaled_qk_T = proj_q.MATMUL_2D(proj_k) / scale;

		Tensor softmax_tensor = softmax(scaled_qk_T);
		Tensor ret = softmax_tensor.MATMUL_2D(proj_v);

		heads.push_back(ret);
	}
	// ok weve finished all of the single head std...
	// now concat and multiply by the BIG matrix Wo
	// and since we concat on the last dim...
	Tensor head_output = concat(heads, heads[0].dim()-1);

	// now mult by the output projection
	Tensor attn_output = output_projection_layer->forward(head_output);
	return attn_output; 

}



Tensor SelfAttnBlock::forward(Tensor t) {
	Tensor n1 = norm1->forward(t);
	Tensor a = attn->forward(n1);
	Tensor x = t + a;
	Tensor n2 = norm2->forward(x);
	Tensor ffn = mlp->forward(n2);
	Tensor y = x + ffn;

	return y;
}

