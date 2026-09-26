#include <iostream>

#include "tensor.h"
#include "linear.h"
#include "layer.h"
#include "train_loop.h"

int main () {
	
	SequentialModel model;
	model.register_layer<Linear>(1, 32);
	//model.register_layer<LinearSigmoid>(32, 32);
	model.register_layer<LinearReLU>(32, 32);
	model.register_layer<LinearReLU>(32, 32);
	//model.register_layer<LinearReLU>(32, 32);
	model.register_layer<Linear>(32, 1);
	//model.register_layer<LinearSigmoid>(300, 1);
	
	train_fn(model);	
	
}
