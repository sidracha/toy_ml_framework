#pragma once

#include "tensor.h"

#include <vector>


// layer also has to implement set the grad to 0
class Layer {
public:
	Layer() {}

	virtual Tensor forward(Tensor t) = 0;
	virtual void zero_grad() = 0;
	virtual void gradient_descent_step(double lr) = 0;
	virtual void clear_graph() = 0;

};

class SequentialModel {
public:
	std::vector<std::unique_ptr<Layer>> layers;

	SequentialModel() {}
	
	template <typename T, typename... Args>
	void register_layer(Args&&... args) {
		layers.push_back(
			std::make_unique<T>(std::forward<Args>(args)...)	
		);
	}
	
	Tensor forward(Tensor t) {
		for (const auto& layer : layers) {
			t = layer->forward(t);
		}
		return t;
	}

	void clear_graph() {
		for (const auto& layer : layers) {
			layer->clear_graph();
		}
	}
		
};
