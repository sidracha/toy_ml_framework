#pragma once 

#include <vector>

#include "tensor.h"

std::vector<double> fourier_initial_condition(int x_size, int k, double L);


Tensor delx_1d_periodic(Tensor pred, double delta_x);
void delx_1d_periodic_backward(TensorNode* ux);

Tensor del2x_1d_periodic(Tensor pred, double delta_x);
void del2x_1d_periodic_backward(TensorNode* uxx);
