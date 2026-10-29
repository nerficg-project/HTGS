#pragma once

#include <torch/extension.h>

namespace htgs::envmap {

    torch::Tensor forward_wrapper(
        const torch::Tensor& sh_coefficients,
        const torch::Tensor& cam_rotation,
        const int total_sh_bases,
        const int active_sh_bases,
        const int width,
        const int height,
        const float focal_x,
        const float focal_y,
        const float center_x,
        const float center_y,
        const bool to_chw);

    torch::Tensor backward_wrapper(
        const torch::Tensor& grad_image,
        const torch::Tensor& cam_rotation,
        const int total_sh_bases,
        const int active_sh_bases,
        const int width,
        const int height,
        const float focal_x,
        const float focal_y,
        const float center_x,
        const float center_y,
        const bool to_chw);

}