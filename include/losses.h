#pragma once

#include "tensor.h"

Tensor MSELoss(Tensor a, Tensor b);
void MSELoss_backward(TensorNode* node);
