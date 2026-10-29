#pragma once

#include <torch/extension.h>
#include <tuple>

namespace htgs::rasterization {

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int>
    forward_wrapper(
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
        const float focal_x,
        const float focal_y,
        const float center_x,
        const float center_y,
        const float near_plane,
        const float far_plane,
        const float scale_modifier,
        const bool use_aaa_mode);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    backward_wrapper(
        torch::Tensor& densification_info,
        const torch::Tensor& grad_image,
        const torch::Tensor& grad_alpha,
        const torch::Tensor& image,
        const torch::Tensor& alpha,
        const torch::Tensor& positions,
        const torch::Tensor& scales,
        const torch::Tensor& rotations,
        const torch::Tensor& opacities,
        const torch::Tensor& sh_rest,
        const torch::Tensor& per_primitive_buffers,
        const torch::Tensor& per_tile_buffers,
        const torch::Tensor& per_instance_buffers,
        const torch::Tensor& per_pixel_buffers,
        const torch::Tensor& M,
        const torch::Tensor& VPM,
        const torch::Tensor& cam_position,
        const int rasterizer_mode,
        const int K,
        const int active_sh_bases,
        const int width,
        const int height,
        const float focal_x,
        const float focal_y,
        const float center_x,
        const float center_y,
        const float near_plane,
        const float far_plane,
        const float scale_modifier,
        const bool use_aaa_mode,
        const int n_instances,
        const int instance_primitive_indices_selector,
        const bool use_distance_scaling);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    inference_wrapper(
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
        const float focal_x,
        const float focal_y,
        const float center_x,
        const float center_y,
        const float near_plane,
        const float far_plane,
        const float scale_modifier,
        const bool use_aaa_mode,
        const bool to_chw,
        const bool use_median_depth);

    void
    update_max_weights_wrapper(
        torch::Tensor& max_weights,
        const torch::Tensor& positions,
        const torch::Tensor& scales,
        const torch::Tensor& rotations,
        const torch::Tensor& opacities,
        const torch::Tensor& M,
        const torch::Tensor& VPM,
        const torch::Tensor& cam_position,
        const int rasterizer_mode,
        const int K,
        const int active_sh_bases,
        const int width,
        const int height,
        const float focal_x,
        const float focal_y,
        const float center_x,
        const float center_y,
        const float near_plane,
        const float far_plane,
        const float scale_modifier,
        const bool use_aaa_mode,
        const float weight_threshold);

}
