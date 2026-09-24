#include <iostream>

#include "tensor.h"
#include "linear.h"
#include "layer.h"
#include "train_loop.h"

int main () {
	
	SequentialModel model;
	model.register_layer<LinearSigmoid>(1, 30);
	model.register_layer<LinearSigmoid>(30, 100);
	model.register_layer<LinearSigmoid>(100, 300);
	model.register_layer<LinearSigmoid>(300, 1);
	
	train_sine(model);	
	
}
