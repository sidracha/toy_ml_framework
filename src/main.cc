#include <iostream>

#include "tensor.h"
#include "linear.h"
#include "layer.h"
#include "train_loop.h"

class MLP : public Model {
public:
	SequentialLayer seq;
	MLP() : Model(), seq() {
		// i guess we put everything in init
		seq.register_layer<LinearSigmoid>(1, 32);
		seq.register_layer<LinearSigmoid>(32, 32);
		seq.register_layer<LinearSigmoid>(32, 32);
		seq.register_layer<Linear>(32, 1);

	}

	Tensor forward(Tensor t) {
		return seq.forward(t);
	}
	
	std::vector<std::unique_ptr<Layer>>& get_layers() {
		return seq.layers;
	}

};

int main () {
	
	MLP mlp; 
	train_fn(mlp);
	return 0;
	
}
