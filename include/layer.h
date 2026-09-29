#pragma once

#include "tensor.h"

#include <vector>


// layer also has to implement set the grad to 0
class Layer {
public:
	std::vector<std::unique_ptr<Layer>> layers;

	Layer() {}

	virtual Tensor forward(Tensor t) = 0;
	virtual void zero_grad() = 0;
	virtual void gradient_descent_step(double lr) = 0;

	virtual std::vector<std::unique_ptr<Layer>>& get_layers() {
		return layers;
	}

};


class SequentialLayer : public Layer {
public:
	SequentialLayer() {}
	
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

	void zero_grad() {
		for (const auto& layer : layers) layer->zero_grad();
	}
	void gradient_descent_step(double lr) {
		for (const auto& layer : layers) layer->gradient_descent_step(lr);
	}

};

class Model : public Layer {
public:
	std::vector<std::unique_ptr<Layer>> layers;
	Model() {};
};
