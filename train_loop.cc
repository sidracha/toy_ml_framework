#include "tensor.h"
#include "optimizer.h"
#include "losses.h"
#include "nn.h"

#include <vector>
#include <cmath>
#include <iostream>

Tensor create_expect_sine(Tensor t) {
	
	// create a new tensor of the same size
	Tensor target = create_tensor_clone_scalar(t, t.graph, 0.0);
	// for each one of tensor do sine
	for (int i=0; i<t.tensor_node->data.size(); i++) {
		target.tensor_node->data[i] = std::sin(t.tensor_node->data[i]);
	}
	return target;

}

// since rows are the batch this will be of [BATCH_SIZE, 1]
Tensor create_validation_set(int batch_size, Graph* graph) {	
	Tensor t = create_tensor_random({batch_size, 1}, graph, -4.0, 4.0);
}

//lets train a simple predictor of sine and lets batch it
void train_sine(SequentialModel& model) {
	
	// weve already created the model and it has a forward method
	// first register a graph to spawn the tensor off of...
	// we can just first create random tensors its fine... they will be our batch sine....
	// and these will be our input target
	
	Graph validation_graph;
	Tensor validation_tensor = create_validation_set(16, validation_graph);

	Graph g;
	double learning_rate = 0.001;
	Optimizer optim(learning_rate, model.layers);
	// now lets create random tenosrs off g in a loop
	// and this is our training loop
	
	int NUM_ITERATIONS = 100;

	while (NUM_ITERATIONS--) {

		Tensor input_tensor = create_tensor_random({16, g, -4.0, 4.0});
		Tensor output_tensor = model.forward(input_tensor);
		Tensor target_tensor = create_expect_sine(input_tensor);


		Tensor loss = MSELoss(output_tensor, target_tensor);
		std::cout << loss.tensor_node->data[0] << std::endl;
		
		optimizer.zero_grad();
		loss.backward();
		
		model.clear_graph();
		g.clear();

	}

}
