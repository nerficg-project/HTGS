#include "fast_inference_api.h"

#include "alpha_blend_first_k/fast_inference.h"
#include "alpha_blend_global_ordering/fast_inference.h"
#include "hybrid_blend/fast_inference.h"
#include "oit_blend/fast_inference.h"

#include "torch_utils.h"
#include "helper_math.h"
#include <torch/extension.h>
#include <stdexcept>
#include <functional>

enum class RasterizerMode {
    HYBRID_BLEND = 0,
    ALPHA_BLEND_FIRST_K = 1,
    ALPHA_BLEND_GLOBAL_ORDERING = 2,
    OIT_BLEND = 3,
};

torch::Tensor htgs::rasterization::fast_inference_wrapper(
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
    const bool to_chw)
{
    const RasterizerMode mode = static_cast<RasterizerMode>(rasterizer_mode);
    const int n_primitives = positions.size(0);
    const int total_sh_bases = sh_rest.size(1);
    const torch::TensorOptions float_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    const torch::TensorOptions byte_options = torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA);
    torch::Tensor image = to_chw ? torch::empty({3, height, width}, float_options) : torch::empty({height, width, 3}, float_options);
    torch::Tensor per_primitive_buffers = torch::empty({0}, byte_options);
    torch::Tensor per_tile_buffers = torch::empty({0}, byte_options);
    torch::Tensor per_instance_buffers = torch::empty({0}, byte_options);
    const std::function<char*(size_t)> per_primitive_buffers_func = resize_function_wrapper(per_primitive_buffers);
    const std::function<char*(size_t)> per_tile_buffers_func = resize_function_wrapper(per_tile_buffers);
    const std::function<char*(size_t)> per_instance_buffers_func = resize_function_wrapper(per_instance_buffers);

    switch (mode) {
        case RasterizerMode::HYBRID_BLEND:
            hybrid_blend::fast_inference(
                per_primitive_buffers_func,
                per_tile_buffers_func,
                per_instance_buffers_func,
                reinterpret_cast<const float3*>(positions.contiguous().data_ptr<float>()),
                reinterpret_cast<const float3*>(scales.contiguous().data_ptr<float>()),
                reinterpret_cast<const float4*>(rotations.contiguous().data_ptr<float>()),
                opacities.contiguous().data_ptr<float>(),
                reinterpret_cast<const float3*>(sh_0.contiguous().data_ptr<float>()),
                reinterpret_cast<const float3*>(sh_rest.contiguous().data_ptr<float>()),
                reinterpret_cast<const float4*>(M.contiguous().data_ptr<float>()),
                reinterpret_cast<const float4*>(VPM.contiguous().data_ptr<float>()),
                reinterpret_cast<const float3*>(cam_position.contiguous().data_ptr<float>()),
                image.data_ptr<float>(),
                K,
                n_primitives,
                active_sh_bases,
                total_sh_bases,
                width,
                height,
                near_plane,
                far_plane,
                scale_modifier,
                to_chw);
            break;
        case RasterizerMode::ALPHA_BLEND_FIRST_K:
            alpha_blend_first_k::fast_inference(
                per_primitive_buffers_func,
                per_tile_buffers_func,
                per_instance_buffers_func,
                reinterpret_cast<const float3*>(positions.contiguous().data_ptr<float>()),
                reinterpret_cast<const float3*>(scales.contiguous().data_ptr<float>()),
                reinterpret_cast<const float4*>(rotations.contiguous().data_ptr<float>()),
                opacities.contiguous().data_ptr<float>(),
                reinterpret_cast<const float3*>(sh_0.contiguous().data_ptr<float>()),
                reinterpret_cast<const float3*>(sh_rest.contiguous().data_ptr<float>()),
                reinterpret_cast<const float4*>(M.contiguous().data_ptr<float>()),
                reinterpret_cast<const float4*>(VPM.contiguous().data_ptr<float>()),
                reinterpret_cast<const float3*>(cam_position.contiguous().data_ptr<float>()),
                image.data_ptr<float>(),
                K,
                n_primitives,
                active_sh_bases,
                total_sh_bases,
                width,
                height,
                near_plane,
                far_plane,
                scale_modifier,
                to_chw);
            break;
        case RasterizerMode::ALPHA_BLEND_GLOBAL_ORDERING:
            alpha_blend_global_ordering::fast_inference(
                per_primitive_buffers_func,
                per_tile_buffers_func,
                per_instance_buffers_func,
                reinterpret_cast<const float3*>(positions.contiguous().data_ptr<float>()),
                reinterpret_cast<const float3*>(scales.contiguous().data_ptr<float>()),
                reinterpret_cast<const float4*>(rotations.contiguous().data_ptr<float>()),
                opacities.contiguous().data_ptr<float>(),
                reinterpret_cast<const float3*>(sh_0.contiguous().data_ptr<float>()),
                reinterpret_cast<const float3*>(sh_rest.contiguous().data_ptr<float>()),
                reinterpret_cast<const float4*>(M.contiguous().data_ptr<float>()),
                reinterpret_cast<const float4*>(VPM.contiguous().data_ptr<float>()),
                reinterpret_cast<const float3*>(cam_position.contiguous().data_ptr<float>()),
                image.data_ptr<float>(),
                n_primitives,
                active_sh_bases,
                total_sh_bases,
                width,
                height,
                near_plane,
                far_plane,
                scale_modifier,
                to_chw);
            break;
        case RasterizerMode::OIT_BLEND:
            oit_blend::fast_inference(
                per_primitive_buffers_func,
                per_tile_buffers_func,
                per_instance_buffers_func,
                reinterpret_cast<const float3*>(positions.contiguous().data_ptr<float>()),
                reinterpret_cast<const float3*>(scales.contiguous().data_ptr<float>()),
                reinterpret_cast<const float4*>(rotations.contiguous().data_ptr<float>()),
                opacities.contiguous().data_ptr<float>(),
                reinterpret_cast<const float3*>(sh_0.contiguous().data_ptr<float>()),
                reinterpret_cast<const float3*>(sh_rest.contiguous().data_ptr<float>()),
                reinterpret_cast<const float4*>(M.contiguous().data_ptr<float>()),
                reinterpret_cast<const float4*>(VPM.contiguous().data_ptr<float>()),
                reinterpret_cast<const float3*>(cam_position.contiguous().data_ptr<float>()),
                image.data_ptr<float>(),
                n_primitives,
                active_sh_bases,
                total_sh_bases,
                width,
                height,
                near_plane,
                far_plane,
                scale_modifier,
                to_chw);
            break;
        default:
            throw std::runtime_error("unsupported rasterizer mode");
    }
    
    return image;
}
