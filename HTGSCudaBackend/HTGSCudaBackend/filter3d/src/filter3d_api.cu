#include "filter3d_api.h"
#include "filter3d.h"
#include "helper_math.h"
#include <functional>

void htgs::filter3d::update_3d_filter_wrapper(
    const torch::Tensor& positions,
    const torch::Tensor& w2c,
    torch::Tensor& filter_3d,
    torch::Tensor& visibility_mask,
    const int width,
    const int height,
    const float focal_x,
    const float focal_y,
    const float center_x,
    const float center_y,
    const float near_plane,
    const float clipping_tolerance,
    const float distance2filter)
{
    const int n_points = positions.size(0);

    const float bounds_factor = clipping_tolerance + 0.5f;
    const float width_f = static_cast<float>(width);
    const float height_f = static_cast<float>(height);
    const float max_x_shifted = bounds_factor * width_f;
    const float max_y_shifted = bounds_factor * height_f;
    const float principal_offset_x = center_x - 0.5f * width_f;
    const float principal_offset_y = center_y - 0.5f * height_f;
    const float left = (-max_x_shifted - principal_offset_x) / focal_x;
    const float right = (max_x_shifted - principal_offset_x) / focal_x;
    const float top = (-max_y_shifted - principal_offset_y) / focal_y;
    const float bottom = (max_y_shifted - principal_offset_y) / focal_y;

    update_3d_filter(
        reinterpret_cast<const float3*>(positions.data_ptr<float>()),
        reinterpret_cast<const float4*>(w2c.data_ptr<float>()),
        filter_3d.data_ptr<float>(),
        visibility_mask.data_ptr<bool>(),
        n_points,
        left,
        right,
        top,
        bottom,
        near_plane,
        distance2filter);
}
