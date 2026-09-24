#include "tensor.h"
#include "layer.h"

// we do SIMPLE gradient descent
// nothing fanciful

// stores a vector list of layers...
// this will just hold the parameters
// can change this later to have a model class

// intermediete nodes are owned by the same Graph object that input is spawned off of...

	
void Optimizer::zero_grad() {
	// go through every single tensor in layers.. and set it to 0
	// but also each layer should have the interface to set to 0
	for (int i=0; i<(*layers).size(); i++) {
		(*layers)[i].zero_grad();
	}
}


// just goes through all the layers and updates their parameters
// very naive
void Optimizer::step() {
	
	for (int i=0; i<(*layers).size(); i++) {
		(*layers)[i].gradient_descent_step(lr);
	}

} 
