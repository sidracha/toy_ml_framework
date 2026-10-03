#pragma once

#include "dataset.h"

#include <vector>

std::vector<double> ks_forward_euler_step_periodic(std::vector<double>& u, double delta_x, double delta_t);
void ks_init_dataset(Dataset& train_dataset, Dataset& test_dataset);
