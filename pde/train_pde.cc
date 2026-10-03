#include <vector>
#include <cmath>
#include <utility>
#include <numbers>
#include <random>
#include <iostream>
#include <optional>

#include "burgers.h"
#include "attention.h"
#include "nn.h"
#include "optimizer.h"
#include "losses.h"
#include "dataset.h"
#include "visualize.h"
#include "pde.h"
#include "ks.h"
#include "kdv.h"

void pde_train_loop() {

	int num_blocks = 4;
	int num_heads = 4;
	int embed_dim = 8;
	int input_embed_dim = 1;
	int output_embed_dim = 1;
	int mlp_ratio = 2;
	auto identity = [](const Tensor& t) { return t; };
	//Transformer model(num_blocks, num_heads, embed_dim, input_embed_dim, output_embed_dim, mlp_ratio, Tanh);
	MLP model(input_embed_dim, output_embed_dim, num_blocks, 32, Tanh);

	double lr = 0.02;
	double lr_multiplier = 0.92;
	Optimizer optimizer(model.layers, lr);

	int x_size = 64;
	double delta_t = 0.02;
	double nu = 0.6;
	double L = 6 * std::numbers::pi;
	double delta_x = L / (double) x_size;

	Dataset train_dataset;
	Dataset test_dataset;
	burgers_init_dataset(train_dataset, test_dataset, L, x_size, delta_t, nu);
	//ks_init_dataset(train_dataset, test_dataset);
	//kdv_init_dataset(train_dataset, test_dataset);

	int batch_size = 4;
	int epoch_size = train_dataset.data_tensors.size() / batch_size;
	int num_epochs = 8;

	double r_scale = 0.2;
	double r_mult;
	Tensor zero_tensor = create_tensor_zeros({batch_size, x_size, 1});

	int autoreg_steps = 6;
	int step = 0;
	bool use_autoregressive = false;

	std::optional<Tensor> input = std::nullopt;
	for (int epoch=0; epoch<num_epochs; epoch++) {
		train_dataset.prepare_epoch(batch_size);
		for (int i=0; i<epoch_size; i++) {
			auto [_, target] = train_dataset.get_input();

			if (use_autoregressive) {
				if (!input.has_value()) input = _;
			} else {
				input = _;
			}

			Tensor pred = model.forward(input.value());

			Tensor data_loss = MSELoss(pred, target);

			Tensor residual = burgers_residual(input.value(), pred, delta_x, delta_t, nu);
			for (int g=0; g<zero_tensor.tensor_node->grad.size(); g++) {
				zero_tensor.tensor_node->grad[g] = 0.0;
			}
			Tensor residual_loss = MSELoss(residual, zero_tensor);
			r_mult = (data_loss.data_at(0) * r_scale) / residual_loss.data_at(0);
			r_mult = std::min(r_mult, 1.0);

			Tensor loss = data_loss + residual_loss * r_mult;

			std::cout << "epoch: " << epoch << " | iteration: " << i << " loss: " << data_loss.data_at(0) << std::endl;

			optimizer.zero_grad();
			loss.backward();
			optimizer.step();
			loss.tensor_node->predecessors.clear();
			residual.tensor_node->predecessors.clear();
			pred.tensor_node->predecessors.clear();

			if (use_autoregressive) {
				if (step >= autoreg_steps) {
					step = 0;
					input = std::nullopt;
				}
				else {
					input = create_tensor(pred.tensor_node->data, pred.shape());
				}
				step++;
			}

		}
		optimizer.lr *= lr_multiplier;
	}

	std::cout << std::endl;
	std::cout << "---------- VALIDATION ----------";
	std::cout << std::endl;
	std::vector<std::vector<double>> preds;
	std::vector<std::vector<double>> targets;

	int test_size = test_dataset.data_tensors.size();

	input = std::nullopt;
	for (int i=0; i<test_size; i++) {
		auto& [_, target] = test_dataset.data_tensors[i];

		if (use_autoregressive) {
			if (!input.has_value()) input = _;
		} else {
			input = _;
		}

		Tensor pred = model.forward(input.value());
		Tensor loss = MSELoss(pred, target);
		std::cout << "iteration: " << i << " loss: " << loss.data_at(0) << std::endl;

		preds.push_back(pred.tensor_node->data);
		targets.push_back(target.tensor_node->data);

		if (use_autoregressive) {
			input = create_tensor(pred.tensor_node->data, pred.shape());
		}
	}

  save_burgers_video(preds, targets, "burgers_comparison");

}
