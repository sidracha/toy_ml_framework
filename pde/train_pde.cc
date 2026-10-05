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

void autoreg_train_loop(
	Layer& model,
	Optimizer& optimizer,
	AutoregDataset& train_dataset,
	int batch_size,
	int num_steps,
	int epoch) {

	train_dataset.prepare_epoch(batch_size, num_steps);

	for (int s=0; s<train_dataset.num_samples(); s++) {
		// get the vector
		std::vector<Tensor> trajectory = train_dataset.get_input();
		// now iterate through
		Tensor input = trajectory[0];

		std::vector<Tensor> losses;

		for (int i=1; i<trajectory.size(); i++) {
			// we want to predict the residual
			Tensor pred_residual = model.forward(input);
			Tensor target_residual = trajectory[i] - input;

			Tensor step_loss = MSELoss(pred_residual, target_residual);
			losses.push_back(step_loss);

			Tensor pred = input + pred_residual;
			input = pred;
		}

		// sum all losses
		Tensor total_loss = losses[0];
		for (int i=1; i<losses.size(); i++) {
			total_loss = total_loss + losses[i];
		}

		std::cout << "epoch: " << epoch << " | sample: " << s << " | total_loss: " << total_loss.data_at(0) << std::endl;

		optimizer.zero_grad();
		total_loss.backward();
		optimizer.step();

		total_loss.tensor_node->predecessors.clear();
	}
}

void autoreg_test_loop(Layer& model, AutoregDataset& test_dataset, int num_steps) {

	test_dataset.prepare_epoch(1, num_steps);

	std::vector<std::vector<double>> preds;
	std::vector<std::vector<double>> targets;

	for (int s=0; s<test_dataset.num_samples(); s++) {
		// get the vector
		std::vector<Tensor> trajectory = test_dataset.get_input();
		// now iterate through
		Tensor input = trajectory[0];
		for (int i=1; i<trajectory.size(); i++) {
			// i is the target

			// we want to predict the residual... so the target=
			Tensor pred_residual = model.forward(input);
			Tensor target = trajectory[i];

			Tensor pred = input + pred_residual;

			preds.push_back(pred.tensor_node->data);
			targets.push_back(target.tensor_node->data);

			input = create_tensor(pred.tensor_node->data, pred.shape());
		}
	}

	save_burgers_video(preds, targets, "burgers_comparison");
}

void pde_train_loop() {

	int x_size = 32;
	int num_blocks = 2;
	int num_heads = 4;
	int embed_dim = 32;
	int input_embed_dim = 1;
	int output_embed_dim = 1;
	int mlp_ratio = 2;
	Transformer model(num_blocks, num_heads, embed_dim, input_embed_dim, output_embed_dim, mlp_ratio, Tanh);

	double lr = 0.001;
	Adam optimizer(model.get_params(), lr);

	double delta_t = 0.05;
	double nu = 0.6;
	double L = 6 * std::numbers::pi;
	double delta_x = L / (double) x_size;

	AutoregDataset train_dataset;
	AutoregDataset test_dataset;
	burgers_init_dataset(train_dataset, test_dataset, L, x_size, delta_t, nu);

	int batch_size = 10;
	int epoch_size = train_dataset.data_tensors.size() / batch_size;
	int num_epochs = 4;

	double r_scale = 0.0;
	double r_mult;
	Tensor zero_tensor = create_tensor_zeros({batch_size, x_size, 1});
	
	int num_autoreg_steps = 10;

	for (int epoch=0; epoch<num_epochs; epoch++) {
		autoreg_train_loop(model, optimizer, train_dataset, batch_size, num_autoreg_steps, epoch);
	}

	autoreg_test_loop(model, test_dataset, 200);

}
