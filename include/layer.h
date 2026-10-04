#pragma once

#include "tensor.h"

#include <vector>


class Layer {
public:
	std::vector<std::unique_ptr<Layer>> layers;
	std::vector<Tensor> params;

	Layer() {}
	virtual ~Layer() = default;

	virtual Tensor forward(Tensor t) = 0;

	virtual std::vector<std::unique_ptr<Layer>>& get_layers() {
		return layers;
	}
	
	std::vector<Tensor*> get_params() { 
		
		std::vector<Tensor*> out;
		
		// through layers first recursively
		for (const auto& layer : layers) {
			std::vector<Tensor*> layer_params = layer->get_params();
			for (int i=0; i<layer_params.size(); i++) out.push_back(layer_params[i]);
		}
		
		// now through the current layers params
		for (int i=0; i<params.size(); i++) {
			out.push_back(&params[i]);
		}
		return out;
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

};

class Model : public Layer {
public:
	std::vector<std::unique_ptr<Layer>> layers;
	Model() {};
};
