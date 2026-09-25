#include "tensor.h"
#include "optimizer.h"
#include "losses.h"
#include "nn.h"

#include <vector>
#include <cmath>
#include <iostream>
#include <matplot/matplot.h>

void plot_sine_prediction(Tensor& input, Tensor& prediction) {
	std::vector<double> x_curve, y_curve;
	for (double x = -3.14; x <= 3.14; x += 0.01) {
		x_curve.push_back(x);
		y_curve.push_back(std::sin(x));
	}

	std::vector<double> x_input, y_pred;
	for (int i = 0; i < input.tensor_node->data.size(); i++) {
		x_input.push_back(input.tensor_node->data[i]);
		y_pred.push_back(prediction.tensor_node->data[i]);
	}

	matplot::plot(x_curve, y_curve, "-b");
	matplot::hold(matplot::on);
	matplot::scatter(x_input, y_pred)->marker_size(5).marker_color("red");
	matplot::hold(matplot::off);
	matplot::show();
}

Tensor create_expect_sine(Tensor t) {
	
	// create a new tensor of the same size
	Tensor target = create_tensor_clone_scalar(t, 0.0);
	// for each one of tensor do sine
	for (int i=0; i<t.tensor_node->data.size(); i++) {
		target.tensor_node->data[i] = std::sin(t.tensor_node->data[i]);
	}
	return target;

}

// since rows are the batch this will be of [BATCH_SIZE, 1]
Tensor create_validation_set(int batch_size) {	
	Tensor t = create_tensor_random({batch_size, 1}, -3.14, 3.14);
	return t;
}

//lets train a simple predictor of sine and lets batch it
void train_sine(SequentialModel& model) {
	
	// weve already created the model and it has a forward method
	// we can just first create random tensors its fine... they will be our batch sine....
	// and these will be our input target
	

	double learning_rate = 0.05;
	Optimizer optim(model.layers, learning_rate);
	// now lets create random tenosrs off g in a loop
	// and this is our training loop
	
	int NUM_ITERATIONS = 2500;

	while (NUM_ITERATIONS--) {

		Tensor input_tensor = create_tensor_random({64, 1}, -3.14, 3.14);
		Tensor output_tensor = model.forward(input_tensor);
		Tensor target_tensor = create_expect_sine(input_tensor);
	
		double output_sum = 0.0;
		for (int i=0; i<output_tensor.tensor_node->data.size(); i++) {
			output_sum += output_tensor.tensor_node->data[i];
		}

		Tensor loss = MSELoss(output_tensor, target_tensor);
		std::cout << loss.tensor_node->data[0] << " " << output_sum / static_cast<double>(64) << std::endl;
		
		optim.zero_grad();
		loss.backward();
		optim.step();
	}
	
	Tensor plot_input_tensor = create_tensor_random({64, 1}, -3.14, 3.14);
	Tensor plot_output_tensor = model.forward(plot_input_tensor);

	plot_sine_prediction(plot_input_tensor, plot_output_tensor);

}
