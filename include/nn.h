#pragma once
#include <vector>

#include "tensor.h"

Tensor ReLU(const Tensor& t);

double sigmoid(double x);
Tensor Sigmoid(const Tensor& t);
void Sigmoid_backward(TensorNode* node);

Tensor softmax(const Tensor& t);
void softmax_backward(TensorNode* node);

Tensor concat(const std::vector<Tensor>& array, int concat_dim);
void concat_backward(TensorNode* node);
