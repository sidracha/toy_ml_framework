#pragma once
#include <vector>

#include "tensor.h"

Tensor ReLU(const Tensor& t);
double sigmoid(double x);
Tensor Sigmoid(const Tensor& t);
void Sigmoid_backward(TensorNode* node);
