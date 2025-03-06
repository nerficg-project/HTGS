#pragma once

#include <torch/extension.h>

namespace htgs::filter3d {

    void update_3d_filter_wrapper(
        const torch::Tensor& positions,
        const torch::Tensor& w2c,
        torch::Tensor& filter_3d,
        torch::Tensor& visibility_mask,
        const uint width,
        const uint height,
        const float focal_x,
        const float focal_y,
        const float principal_offset_x,
        const float principal_offset_y,
        const float near_plane,
        const float clipping_tolerance,
        const float distance2filter);

}
