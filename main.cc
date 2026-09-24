#include <iostream>

#include "tensor.h"
#include "linear.h"
#include "layer.h"
#include "train_loop.h"

int main () {
	
	SequentialModel model;
	model.register_layer<LinearReLU>(1, 30);
	model.register_layer<LinearReLU>(30, 100);
	model.register_layer<LinearReLU>(100, 300);
	model.register_layer<Linear>(300, 1);
	
	train_sine(model);	
	
}
