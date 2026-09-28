#include "tensor.h"
#include "layer.h"
#include "linear.h"

class SelfAttn : public Layer {
public:
	std::vector<std::unique_ptr<Layer>> layers;
	int embed_dim;
	int hidden_dim;
		
	Linear* wq;
	Linear* wk;
	Linear* wv;

	SelfAttn(int _embed_dim, int _hidden_dim) : 
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

	Tensor forward(Tensor t) {
		
		// ok do the qkv projections here i guess
		Tensor proj_q = wq->forward(t);
		Tensor proj_k = wq->forward(t);
		Tensor proj_v = wq->forward(t);
	
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

	void zero_grad() {
		for (int i=0; i<layers.size(); i++) layers[i]->zero_grad();	
	}

	void gradient_descent_step(double lr) {
		for (int i=0; i<layers.size(); i++) layers[i]->gradient_descent_step(lr);
	}

};
