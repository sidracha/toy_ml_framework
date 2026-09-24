#include <iostream>

#include "tensor.h"
#include "linear.h"
#include "layer.h"
#include "train_loop.h"

int main () {
	
	SequentialModel model;
	model.register_layer<LinearSigmoid>(1, 100);
	model.register_layer<LinearSigmoid>(100, 15);
	model.register_layer<Linear>(15, 1);
	//model.register_layer<LinearSigmoid>(300, 1);
	
	train_sine(model);	
	
}
