#include "tensor.h"
#include "optimizer.h"
#include "losses.h"
#include "nn.h"
#include "trainers.h"
#include "linear.h"
#include "attention.h"

#include <vector>
#include <cmath>
#include <iostream>
#include <matplot/matplot.h>

#define PI 3.14
#define PI2 6.28
#define TRAIN_DATA_MIN -3
#define TRAIN_DATA_MAX 3

#define VAL_DATA_MIN -6
#define VAL_DATA_MAX 6

#define B 2
#define S 8
#define E 16

double random_parabola(double x) {
	double y = -pow((x-2.0), 2) + 5.0;
	return y;
}

test_fn_type test_fn = [](double x) {
	return 3*x + 5;
	//return std::sin(x);
	//return random_parabola(x);
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

Tensor create_reversed_last_dim(Tensor t) {
	Tensor target = create_tensor_clone_scalar(t, 0.0);
	std::vector<int> shape = t.shape();
	std::vector<int> stride = t.stride();
	int dim = t.dim();
	int last_dim_size = shape[dim - 1];
	int last_dim_stride = stride[dim - 1];

	int num_slices = 1;
	for (int i = 0; i < dim - 1; i++) num_slices *= shape[i];

	int slice_size = last_dim_size;
	for (int s = 0; s < num_slices; s++) {
		int base = s * slice_size;
		for (int i = 0; i < last_dim_size; i++) {
			int src_idx = base + i;
			int dst_idx = base + (last_dim_size - 1 - i);
			target.tensor_node->data[dst_idx] = t.tensor_node->data[src_idx];
		}
	}
	return target;
}

// since rows are the batch this will be of [BATCH_SIZE, 1]
Tensor create_input_random() {
	
	// introduce some stuff out of dist randomly so we can see if it generealizses
	int x = rand() % 10;
	if (x >= 5) return create_tensor_random({B, S, E}, -3, 0);
	//else if (x >= 6) return create_tensor_random({batch_size, 1}, TRAIN_DATA_MIN+6, TRAIN_DATA_MAX+4);
	else return create_tensor_random({B, S, E}, TRAIN_DATA_MIN, TRAIN_DATA_MAX);
}

void train_fn() {
	
	SelfAttnBlock model(2, E, 4, ReLU);

	double learning_rate = 0.05;
	double learning_rate_multiplier = 0.9996;
	SGDOptimizer optim(model.get_params(), learning_rate);

	int NUM_ITERATIONS = 50000;

	for (int i=0; i<NUM_ITERATIONS; i++) {
		Tensor input_tensor = create_input_random();
		Tensor output_tensor = model.forward(input_tensor);
		Tensor target_tensor = create_reversed_last_dim(input_tensor);

		double output_sum = 0.0;
		for (int i=0; i<output_tensor.tensor_node->data.size(); i++) {
			output_sum += output_tensor.tensor_node->data[i];
		}

		Tensor loss = MSELoss(output_tensor, target_tensor);
		std::cout << i << ": " << loss.tensor_node->data[0] << " " << output_sum / static_cast<double>(B) << std::endl;

		optim.zero_grad();
		loss.backward(false);
		//std::cout << "grad[0]: " << lin->weight.tensor_node->grad[0] << std::endl;
		optim.step();
		learning_rate *= learning_rate_multiplier;
	}
}
