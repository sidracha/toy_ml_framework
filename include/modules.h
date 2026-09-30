#include "tensor.h"
#include "layer.h"
#include "nn.h"

void layer_norm_backward(TensorNode* Y);

class LayerNorm : public Layer {
public:
	// i guess the layers will hold
	// a Layer& object of a single parameter
	// ok so we can have single parameters... no issue
	Tensor gamma;
	Tensor beta;
	
	std::vector<int> normalize_dims;
	double eps = 0.00001;
	
	// ok so apparently gamma is supposed to be init to 1
	// and beta is supposed to be init to 0 so whatever
	LayerNorm(std::vector<int> _normalize_dims, std::vector<int> _dim_sizes) :
		normalize_dims(_normalize_dims),
		gamma(create_tensor_scalar(_dim_sizes, 1.0)),
		beta(create_tensor_scalar(_dim_sizes, 0.0)) {}
	
	Tensor forward(Tensor t);

	void zero_grad() override {
		std::fill(gamma.tensor_node->grad.begin(), gamma.tensor_node->grad.end(), 0.0);
		std::fill(beta.tensor_node->grad.begin(), beta.tensor_node->grad.end(), 0.0);
	}

	void gradient_descent_step(double lr) override {
		for (int i = 0; i < gamma.tensor_node->data.size(); i++) {
			gamma.tensor_node->data[i] -= lr * gamma.tensor_node->grad[i];
			beta.tensor_node->data[i] -= lr * beta.tensor_node->grad[i];
		}
	}
};
