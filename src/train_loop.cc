#include "tensor.h"
#include "optimizer.h"
#include "losses.h"
#include "nn.h"
#include "train_loop.h"

#include <vector>
#include <cmath>
#include <iostream>
#include <matplot/matplot.h>

#define PI 3.14
#define PI2 6.28
#define TRAIN_DATA_MIN 1
#define TRAIN_DATA_MAX 5

#define VAL_DATA_MIN -4
#define VAL_DATA_MAX 4

double random_parabola(double x) {
	double y = -pow((x-2.0), 2) + 5.0;
	return y;
}

test_fn_type test_fn = [](double x) {
	//return std::exp(x);
	return random_parabola(x);
};

void plot_fn_prediction(Tensor& input, Tensor& prediction, test_fn_type fn) {
	std::vector<double> x_curve, y_curve;
	for (double x = VAL_DATA_MIN; x <= VAL_DATA_MAX; x += 0.01) {
		x_curve.push_back(x);
		y_curve.push_back(fn(x));
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

Tensor create_expect_fn(Tensor t, test_fn_type fn) {
	
	// create a new tensor of the same size
	Tensor target = create_tensor_clone_scalar(t, 0.0);
	// for each one of tensor do fn
	for (int i=0; i<t.tensor_node->data.size(); i++) {
		target.tensor_node->data[i] = fn(t.tensor_node->data[i]);
	}
	return target;

}

// since rows are the batch this will be of [BATCH_SIZE, 1]
Tensor create_input_random(int batch_size) {
	
	// introduce some stuff out of dist randomly so we can see if it generealizses
	int x = rand() % 10;
	if (x >= 5) return create_tensor_random({batch_size, 1, 1}, -3, 0);
	//else if (x >= 6) return create_tensor_random({batch_size, 1}, TRAIN_DATA_MIN+6, TRAIN_DATA_MAX+4);
	else return create_tensor_random({batch_size, 1, 1}, 1, 3);
}

//lets train a simple predictor of sine and lets batch it
void train_fn(SequentialModel& model) {
	
	// weve already created the model and it has a forward method
	// we can just first create random tensors its fine... they will be our batch fn....
	// and these will be our input target
	

	double learning_rate = 0.1;
	double learning_rate_multiplier = 0.9996;
	Optimizer optim(model.layers, learning_rate);
	// now lets create random tenosrs off g in a loop
	// and this is our training loop
	
	int NUM_ITERATIONS = 50000;

	while (NUM_ITERATIONS--) {
		
		Tensor input_tensor = create_input_random(64);
		Tensor output_tensor = model.forward(input_tensor);
		Tensor target_tensor = create_expect_fn(input_tensor, test_fn);
	
		double output_sum = 0.0;
		for (int i=0; i<output_tensor.tensor_node->data.size(); i++) {
			output_sum += output_tensor.tensor_node->data[i];
		}

		Tensor loss = MSELoss(output_tensor, target_tensor);
		std::cout << loss.tensor_node->data[0] << " " << output_sum / static_cast<double>(64) << std::endl;
		
		optim.zero_grad();
		loss.backward();
		optim.step();
		learning_rate *= learning_rate_multiplier;
	}
	
	Tensor plot_input_tensor = create_tensor_random({64, 1, 1}, VAL_DATA_MIN, VAL_DATA_MAX);
	Tensor plot_output_tensor = model.forward(plot_input_tensor);

	plot_fn_prediction(plot_input_tensor, plot_output_tensor, test_fn);

}
