#pragma once
#include <vector>

std::vector<double> burgers_fourier_initial_condition(int x_size, int k, double L);
std::vector<double> burgers_forward_euler_step_periodic(std::vector<double>& u_prev, double delta_x, double delta_t, double nu);
void burgers_train_loop();
