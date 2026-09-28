#include "attention.h"

SelfAttn::SelfAttn(int _embed_dim, int _hidden_dim) :
	embed_dim(_embed_dim), hidden_dim(_hidden_dim) {

	// add the linear layers...
	// ohh hmmm we want to embed the embedding_dim
	// into hidden_dim
	layers.push_back(std::make_unique<Linear>(embed_dim, hidden_dim));
	layers.push_back(std::make_unique<Linear>(embed_dim, hidden_dim));
	layers.push_back(std::make_unique<Linear>(embed_dim, hidden_dim));

	// uhh ok so we want to embed each of htem then do the matmuls and the
	// transposes and shit
	wq = dynamic_cast<Linear*>(layers[0].get());
	wk = dynamic_cast<Linear*>(layers[1].get());
	wv = dynamic_cast<Linear*>(layers[2].get());

}

Tensor SelfAttn::forward(Tensor t) {

	// ok do the qkv projections here i guess
	Tensor proj_q = wq->forward(t);
	Tensor proj_k = wk->forward(t);
	Tensor proj_v = wv->forward(t);

	// transpose proj k
	proj_k.transpose();
	double scale = std::pow(hidden_dim, 0.5);
	Tensor scaled_qk_T = proj_q.MATMUL_2D(proj_k) / scale;

	// now we have to do softmax and mult which we only have basically a mock of
	// but whatever
	Tensor softmax_tensor = softmax(scaled_qk_T);
	Tensor ret = softmax_tensor.MATMUL_2D(proj_v);

	return ret;

}

void SelfAttn::zero_grad() {
	for (int i=0; i<layers.size(); i++) layers[i]->zero_grad();
}

void SelfAttn::gradient_descent_step(double lr) {
	for (int i=0; i<layers.size(); i++) layers[i]->gradient_descent_step(lr);
}



