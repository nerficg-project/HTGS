#pragma once

#include <torch/extension.h>

namespace htgs::rasterization {

    torch::Tensor fast_inference_wrapper(
        const torch::Tensor& positions,
        const torch::Tensor& scales,
        const torch::Tensor& rotations,
        const torch::Tensor& opacities,
        const torch::Tensor& sh_0,
        const torch::Tensor& sh_rest,
        const torch::Tensor& M,
        const torch::Tensor& VPM,
        const torch::Tensor& cam_position,
        const int rasterizer_mode,
        const int K,
        const int active_sh_bases,
        const int width,
        const int height,
        const float near_plane,
        const float far_plane,
        const float scale_modifier,
        const bool to_chw);

}
