#pragma once

#include <vector>
#include <string>

void save_burgers_video(
    const std::vector<std::vector<double>>& pred_timesteps,
    const std::vector<std::vector<double>>& target_timesteps,
    const std::string& filename = "burgers_comparison"
);
