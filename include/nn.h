#pragma once
#include <vector>

#include "tensor.h"

Tensor ReLU(const Tensor& t);

double sigmoid(double x);
Tensor Sigmoid(const Tensor& t);
void Sigmoid_backward(TensorNode* node);

Tensor softmax(const Tensor& t);
void softmax_backward(TensorNode* node);

Tensor concat(const std::vector<Tensor>& array, int concat_dim, bool use_grad=true);
void concat_backward(TensorNode* node);

Tensor fixed_sincos_pos_embed(const Tensor& t, int embed_dim);

double tanh(double x);
Tensor Tanh(const Tensor& t);
void Tanh_backward(TensorNode* node);
