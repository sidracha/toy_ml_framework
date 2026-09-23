#pragma once

#include <vector>

#include "tensor.h"

Tensor* ReLU(Tensor* a);
std::vector<Tensor*>* ReLU(std::vector<Tensor*>* a);
