#pragma once
#include <vector>

#include "dataset.h"

std::vector<double> burgers_forward_euler_step_periodic(std::vector<double>& u, double delta_x, double delta_t, double nu);
void burgers_init_dataset(Dataset& train_dataset, Dataset& test_dataset);
