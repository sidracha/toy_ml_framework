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

void pde_train_loop() {	
	int num_blocks = 4;
	int num_heads = 4;
	int embed_dim = 8;
	int input_embed_dim = 1;
	int output_embed_dim = 1;
	int mlp_ratio = 2;
	auto identity = [](const Tensor& t) { return t; };
	Transformer model(num_blocks, num_heads, embed_dim, input_embed_dim, output_embed_dim, mlp_ratio, Tanh);
	
	double lr = 0.08;
	double lr_multiplier = 0.92; 
	Optimizer optimizer(model.layers, lr);

	Dataset train_dataset;
	Dataset test_dataset;
	//burgers_init_dataset(train_dataset, test_dataset);
	ks_init_dataset(train_dataset, test_dataset);

	// now do the training loop
	// can we even tell how many epochs there are? 
	int batch_size = 2;
	int epoch_size = train_dataset.data_tensors.size() / batch_size;
	int num_epochs = 12;
	
	// train loop
	for (int epoch=0; epoch<num_epochs; epoch++) {
		train_dataset.prepare_epoch(batch_size);
		for (int i=0; i<epoch_size; i++) {
			auto [input, target] = train_dataset.get_input();

			Tensor pred = model.forward(input);

			Tensor loss = MSELoss(pred, target);
			std::cout << "epoch: " << epoch << " | iteration: " << i << " loss: " << loss.data_at(0) << std::endl;
			
			optimizer.zero_grad();
			loss.backward();
			optimizer.step();

		}
		optimizer.lr *= lr_multiplier;
	}
	
	std::cout << std::endl;
	std::cout << "---------- VALIDATION ----------";
	std::cout << std::endl;
	std::vector<std::vector<double>> preds;
	std::vector<std::vector<double>> targets;
	
	int test_size = test_dataset.data_tensors.size();
	//std::optional<Tensor> input = std::nullopt;
	for (int i=0; i<test_size; i++) {
		auto& [input, target] = test_dataset.data_tensors[i];
		//if (!input.has_value()) input = _; 

		Tensor pred = model.forward(input);
		Tensor loss = MSELoss(pred, target);
		std::cout << "iteration: " << i << " loss: " << loss.data_at(0) << std::endl;
	
		preds.push_back(pred.tensor_node->data);
		targets.push_back(target.tensor_node->data);
		//input = pred;
	}

  save_burgers_video(preds, targets, "burgers_comparison");

}
