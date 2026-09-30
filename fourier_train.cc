#include "tensor.h"
#include "optimizer.h"
#include "losses.h"
#include "nn.h"
#include "trainers.h"
#include "linear.h"
#include "attention.h"

#include <iostream>
#include <vector>
#include <array>
#include <fftw3.h>
#include <random>
#include <cmath>
#include <matplot/matplot.h>

#define B 4
#define NUM_FREQS 5

double signal_fn(double t, const std::vector<double>& freqs, const std::vector<double>& amps) {
	double val = 0.0;
	for (int i = 0; i < freqs.size(); i++) {
		val += amps[i] * std::sin(freqs[i] * t);
	}
	return val;
}

void plot_output_target(const Tensor& output, const Tensor& target, int N, const std::vector<double>& loss_history) {
	std::vector<int> stride = output.stride();

	std::vector<std::vector<double>> output_mag(B, std::vector<double>(N));
	std::vector<std::vector<double>> target_mag(B, std::vector<double>(N));

	double max_val = 0.0;
	for (int b = 0; b < B; b++) {
		for (int i = 0; i < N; i++) {
			int idx_real = b * stride[0] + i * stride[1];
			int idx_imag = idx_real + 1;

			double o_re = output.tensor_node->data[idx_real];
			double o_im = output.tensor_node->data[idx_imag];
			output_mag[b][i] = std::sqrt(o_re * o_re + o_im * o_im);

			double t_re = target.tensor_node->data[idx_real];
			double t_im = target.tensor_node->data[idx_imag];
			target_mag[b][i] = std::sqrt(t_re * t_re + t_im * t_im);

			max_val = std::max(max_val, std::max(output_mag[b][i], target_mag[b][i]));
		}
	}

	auto fig = matplot::figure();
	fig->size(1500, 400);

	matplot::subplot(1, 3, 0);
	matplot::imagesc(output_mag);
	matplot::colorbar();
	matplot::caxis({0.0, max_val});
	matplot::title("Output");
	matplot::xlabel("Frequency");
	matplot::ylabel("Batch");

	matplot::subplot(1, 3, 1);
	matplot::imagesc(target_mag);
	matplot::colorbar();
	matplot::caxis({0.0, max_val});
	matplot::title("Target");
	matplot::xlabel("Frequency");
	matplot::ylabel("Batch");

	matplot::subplot(1, 3, 2);
	matplot::plot(loss_history);
	matplot::title("Loss");
	matplot::xlabel("Iteration");
	matplot::ylabel("MSE");

	matplot::show();
}

// basically create the target for a FFT
// it will have 2 in the later dim because its basically....
// real, complex of the fft
// creates of size {B, N/2+1, 2}

std::vector<double> create_sample_fn(int N, double MIN, double MAX, std::function<double(double)> fn) {
	double step = (MAX-MIN)/N;
	std::vector<double> out(N);
	for (int i=0; i<N; i++) {
		double x = MIN + i * step;
		out[i] = fn(x);
	}
	return out;
}

Tensor create_input_tensor(int N) {

	std::vector<int> shape = {B, N, 1};
	std::vector<int> stride = {N, 1, 1};

	std::vector<double> data(shape[0] * shape[1] * shape[2]);

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> freq_dist(0.5, N/2 - 0.5);
	std::uniform_real_distribution<double> amp_dist(0.5, 1.5);

	for (int b=0; b<B; b++) {

		std::vector<double> freqs(NUM_FREQS);
		std::vector<double> amps(NUM_FREQS);
		for (int f = 0; f < NUM_FREQS; f++) {
			freqs[f] = freq_dist(gen);
			amps[f] = amp_dist(gen);
		}

		double step = (2.0 * M_PI) / N;
		for (int i=0; i<N; i++) {
			int index = b*stride[0] + i*stride[1];
			data[index] = signal_fn(i * step, freqs, amps);
		}

	}

	std::shared_ptr<TensorNode> node = make_operator_output_node(
		data, shape, {}, nullptr);
	
	return Tensor(node);
}

Tensor create_fft_target_tensor(const Tensor& t, int N) {
	
	// for each batch make a one vector * 2 of it
	
	// the shape of the output tensor is different....
	std::vector<int> output_shape = {B, N, 2};
	std::vector<int> output_stride = {N*2, 2, 1};
	
	std::vector<double> data(output_shape[0] * output_shape[1] * output_shape[2]);
	std::vector<int> input_stride = t.stride();

	for (int b=0; b<B; b++) {
	

		// copy the in from the things...
		fftw_complex* in = fftw_alloc_complex(N);
		fftw_complex* out = fftw_alloc_complex(N);
		for (int i=0; i<N; i++) {
			int input_index = b*input_stride[0] + i*input_stride[1];
			in[i][0] = t.tensor_node->data[input_index];
			in[i][1] = 0.0;
		}

		fftw_plan plan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
		fftw_execute(plan);
		fftw_destroy_plan(plan);

		for (int i=0; i<N; i++) {
			int index_real = b*output_stride[0] + i*output_stride[1];
			int index_imag = b*output_stride[0] + i*output_stride[1] + 1;

			data[index_real] = out[i][0] / N;
			data[index_imag] = out[i][1] / N;
		}

		fftw_free(in);
		fftw_free(out);

	}
	// okk just put it into a tensor then
	std::shared_ptr<TensorNode> node = make_operator_output_node(
		data, output_shape, {}, nullptr);
	
	return Tensor(node);
	

}

void fft_train_loop() {

	int N = 16;
	// so we will have some shape of 
	// {B, N, 2}
	// which will go into our transformer
	int num_blocks = 8;
	int num_heads = 2;
	int embed_dim = 8;
	int input_embed_dim = 1;
	int output_embed_dim = 2;
	int mlp_ratio = 2;
	auto identity = [](const Tensor& t) { return t; };
	Transformer model(num_blocks, num_heads, embed_dim, input_embed_dim, output_embed_dim, mlp_ratio, Tanh);
	
	double lr = 0.7;
	double lr_scale = 0.99998;
	Optimizer optimizer(model.layers, lr);

	int NUM_ITERATIONS = 100000;
	int LOG_INTERVAL = NUM_ITERATIONS / 100;
	std::vector<double> loss_history;

	for (int i=0; i<NUM_ITERATIONS; i++) {
		Tensor input_tensor = create_input_tensor(N);
		Tensor target_tensor = create_fft_target_tensor(input_tensor, N);

		Tensor output_tensor = model.forward(input_tensor);

		double output_sum = 0.0;
		for (int j=0; j<output_tensor.tensor_node->data.size(); j++) {
			output_sum += output_tensor.tensor_node->data[j];
		}

		Tensor loss = MSELoss(output_tensor, target_tensor);
		std::cout << i << ": " << loss.tensor_node->data[0] << " " << output_sum / static_cast<double>(B) << std::endl;
		if (i % LOG_INTERVAL == 0) {
			loss_history.push_back(loss.tensor_node->data[0]);
		}

		optimizer.zero_grad();
		loss.backward(false);
		optimizer.step();
		optimizer.lr *= lr_scale;

	}

	Tensor test_input = create_input_tensor(N);
	Tensor test_target = create_fft_target_tensor(test_input, N);
	Tensor test_output = model.forward(test_input);
	plot_output_target(test_output, test_target, N, loss_history);

}
