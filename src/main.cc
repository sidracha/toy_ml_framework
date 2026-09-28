#include <iostream>

#include "tensor.h"
#include "linear.h"
#include "layer.h"
#include "train_loop.h"

class MLP : public Model {
public:
	std::vector<std::unique_ptr<Layer>> layers;
	//SequentialLayer seq;
	MLP() : Model() {
		// i guess we put everything in init
		/*
		seq.register_layer<LinearSigmoid>(1, 32);
		seq.register_layer<LinearSigmoid>(32, 32);
		seq.register_layer<LinearSigmoid>(32, 32);
		seq.register_layer<Linear>(32, 1);
		*/
		
		layers.push_back(std::make_unique<LinearSigmoid>(1, 32));
		layers.push_back(std::make_unique<LinearSigmoid>(32, 32));
		layers.push_back(std::make_unique<LinearSigmoid>(32, 32));
		layers.push_back(std::make_unique<Linear>(32, 1));

	}

	Tensor forward(Tensor t) {
		t = layers[0]->forward(t);
		Tensor x = t + t;
		x = layers[2]->forward(x);
		x = layers[3]->forward(x);
		return x;
	}
	
	std::vector<std::unique_ptr<Layer>>& get_layers() {
		return layers;
	}

};

int main () {
	
	MLP mlp; 
	train_fn(mlp);
	return 0;
	
}
