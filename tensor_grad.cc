// The problem:
//
// We need a Tensor class that stores value and gradient
// The tensors that produced it, and enough information to propogate the gradients backwards

// first implement addition and multiplication


// example equation:
// d = a + b*c
// we need to find d/da, d/db, d/dc


// so we have (b*c) -> tensor T
// then a + T -> d


// so the graph goes
//
// we have X = ABC
// then T = AB
// dT/dA = B
// dT/dB = A

// X = TC
//
// dX/dT = C
// dX/dT



// a node can only have 2 predecessors.
// when we create a node, we want to add to the predecessors of the new node that 
// is created

#include <vector>
#include <deque>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <random>

#include "tensor.h"
#include "nn.h"
#include "losses.h"
#include "linear.h"

class Optimizer {
public:
	std::vector<Linear>& layers;

	Optimizer(std::vector<Linear>& _layers) : layers(_layers) {};
	
	void zero_grad() {
		// set everything in the graph to 0.
		// do we traverse the graph, and then set everything to 0...
		// but all the intemediate ones should also be 0 so its fine...
		for (Linear& layer : layers) {
		

		}
	}

};


void training_loop() {
	
	// we have one input, which is x. we can create our training set. lets try and predict sin(x) as best as possible and try to interpolate the rest of the points	
	// lets first set up our model, which is 3 layers of linear with a ReLU in the middle
	
	std::vector<Linear> layers = {Linear(1, 50), Linear(50, 100), Linear(100, 50), Linear(50, 1)};
	
	Tensor* x = new Tensor(2.0);
	std::vector<Tensor*>* input = new std::vector<Tensor*>(1);
	(*input)[0] = x;
	
	for (int i=0; i<layers.size(); i++) {
		// call the layer forward method
		input = layers[i].forward(input);
		input = ReLU(input);
			
	}
	
	Tensor* target = new Tensor(1.0);
	// we have the input here, now we have to calculate the loss against the target, and call a backward pass
	Tensor* loss = MSELoss((*input)[0], target);
	loss->backward();

	std::cout << (*input)[0]->value << std::endl;
		
}
