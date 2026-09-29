#include <iostream>

#include "tensor.h"
#include "linear.h"
#include "layer.h"
#include "train_loop.h"
#include "optimizer.h"
#include "attention.h"
#include "losses.h"
#include "nn.h"


int main () {
	
	SelfAttn self_attn(8, 128);

	double learning_rate = 0.05;
	Optimizer optimizer(self_attn.layers, learning_rate);
	// okkk so what do we do here
	// [B, S, E]
	Tensor t1 = create_tensor_zeros({8, 32, 128});
	Tensor t2 = create_tensor_zeros({8, 32, 128});
	Tensor t = concat({t1, t2}, 1);
	Tensor target = create_tensor_scalar({8, 64, 128}, 1.0);


	Tensor output = self_attn.forward(t);
	Tensor loss = MSELoss(output, target);
	

	optimizer.zero_grad();
	loss.backward(false);
	optimizer.step();

	MLP mlp(1, 1, 4, 32, Sigmoid); 
	train_fn(mlp);
	return 0;
	
}
