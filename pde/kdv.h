#pragma once

#include "dataset.h"

#include <vector>

std::vector<double> kdv_forward_euler_step_periodic(std::vector<double>& u, double delta_x, double delta_t, double delta);
void kdv_init_dataset(Dataset& train_dataset, Dataset& test_dataset);
