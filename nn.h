#pragma once
#include <vector>

#include "tensor.h"

Tensor ReLU(Tensor t);
double sigmoid(double x);
Tensor Sigmoid(Tensor t);
void Sigmoid_backward(TensorNode* node);
