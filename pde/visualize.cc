#include "visualize.h"
#include <matplot/matplot.h>
#include <cmath>
#include <filesystem>

void save_burgers_video(
    const std::vector<std::vector<double>>& pred_timesteps,
    const std::vector<std::vector<double>>& target_timesteps,
    const std::string& filename
) {
    using namespace matplot;

    std::string tmp_dir = "/tmp/burgers_frames";
    std::filesystem::create_directories(tmp_dir);

    auto f = figure(true);
    f->size(900, 650);
    f->position({0, 0, 900, 650});

    int num_timesteps = pred_timesteps.size();
    int num_points = pred_timesteps[0].size();

    std::vector<double> x(num_points);
    for (int i = 0; i < num_points; i++) {
        x[i] = static_cast<double>(i);
    }

    double y_min = 0, y_max = 0;
    for (int t = 0; t < num_timesteps; t++) {
        for (int i = 0; i < num_points; i++) {
            y_min = std::min(y_min, std::min(pred_timesteps[t][i], target_timesteps[t][i]));
            y_max = std::max(y_max, std::max(pred_timesteps[t][i], target_timesteps[t][i]));
        }
    }
    double margin = (y_max - y_min) * 0.1;
    y_min -= margin;
    y_max += margin;

    for (int t = 0; t < num_timesteps; t++) {
        cla();
        hold(on);

        auto p1 = plot(x, pred_timesteps[t]);
        p1->line_width(2.5);
        p1->color("red");
        p1->display_name("Prediction");

        auto p2 = plot(x, target_timesteps[t]);
        p2->line_width(2.5);
        p2->color("blue");
        p2->display_name("Target");

        xlim({0, static_cast<double>(num_points)});
        ylim({y_min, y_max});
        xlabel("x");
        ylabel("u");
        title("1D Burgers: ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²   |   Timestep " + std::to_string(t));
        legend();

        std::string frame_filename = tmp_dir + "/frame_" + std::to_string(t) + ".png";
        save(frame_filename);

        hold(off);
    }

    std::string palette_cmd = "ffmpeg -y -framerate 10 -i " + tmp_dir + "/frame_%d.png -vf palettegen " + tmp_dir + "/palette.png 2>/dev/null";
    system(palette_cmd.c_str());

    std::string gif_cmd = "ffmpeg -y -framerate 10 -i " + tmp_dir + "/frame_%d.png -i " + tmp_dir + "/palette.png -lavfi paletteuse -loop 0 " + filename + ".gif 2>/dev/null";
    system(gif_cmd.c_str());

    std::filesystem::remove_all(tmp_dir);
}
