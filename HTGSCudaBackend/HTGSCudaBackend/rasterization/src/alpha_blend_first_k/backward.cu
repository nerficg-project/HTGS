#include "alpha_blend_first_k/backward.h"
#include "alpha_blend_first_k/kernels/backward.cuh"
#include "alpha_blend_first_k/buffer_utils.h"
#include "alpha_blend_first_k/config.h"
#include "rasterization_utils.h"
#include "utils.h"
#include "helper_math.h"
#include <variant>

// the blend kernel is templated on both the AAA mode and the core size, so the instantiation is
// selected as a kernel pointer instead of launching from inside the dispatch
template <bool aaa_mode>
auto select_blend_k_kernel(const int K) {
    if (K >= 32) return htgs::rasterization::alpha_blend_first_k::kernels::backward::blend_cu<aaa_mode, 32>;
    else if (K >= 16) return htgs::rasterization::alpha_blend_first_k::kernels::backward::blend_cu<aaa_mode, 16>;
    else if (K >= 8) return htgs::rasterization::alpha_blend_first_k::kernels::backward::blend_cu<aaa_mode, 8>;
    else if (K >= 4) return htgs::rasterization::alpha_blend_first_k::kernels::backward::blend_cu<aaa_mode, 4>;
    else if (K >= 2) return htgs::rasterization::alpha_blend_first_k::kernels::backward::blend_cu<aaa_mode, 2>;
    else return htgs::rasterization::alpha_blend_first_k::kernels::backward::blend_cu<aaa_mode, 1>;
}

void htgs::rasterization::alpha_blend_first_k::backward(
    const float* grad_image,
    const float* grad_alpha,
    const float* alpha,
    const float3* positions,
    const float3* scales,
    const float4* rotations,
    const float* opacities,
    const float3* sh_rest,
    const float4* M,
    const float4* VPM,
    const float3* cam_position,
    char* per_primitive_buffers_blob,
    char* per_tile_buffers_blob,
    char* per_instance_buffers_blob,
    char* per_pixel_buffers_blob,
    float3* grad_positions,
    float3* grad_scales,
    float4* grad_rotations,
    float* grad_opacities,
    float3* grad_sh_0,
    float3* grad_sh_rest,
    float* densification_info,
    float* densification_info_helper,
    const int K,
    const int n_primitives,
    const int active_sh_bases,
    const int total_sh_bases,
    const int width,
    const int height,
    const int n_instances,
    const int instance_primitive_indices_selector,
    const float max_focal,
    const bool use_aaa_mode,
    const bool use_distance_scaling)
{
    cudaMemcpyToSymbol(c_M, M, 3 * sizeof(float4), 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(c_VPM, VPM, 4 * sizeof(float4), 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(c_cam_position, cam_position, sizeof(float3), 0, cudaMemcpyDeviceToDevice);

    const int n_tiles = div_round_up(width, config::tile_width) * div_round_up(height, config::tile_height);
    const int end_bit = extract_end_bit(n_tiles - 1);
    const int n_pixels = width * height;
    const int n_fragments = K * n_pixels;

    // must match the flag the forward pass sized the blob with, or every pointer carved after the
    // opacity slot shifts
    constexpr bool store_rgb = true, store_rgb_clamp_info = true;
    const bool store_opacity = use_aaa_mode;
    PerPrimitiveBuffers per_primitive_buffers = PerPrimitiveBuffers::from_blob(per_primitive_buffers_blob, n_primitives, store_rgb, store_rgb_clamp_info, store_opacity);
    PerTileBuffers per_tile_buffers = PerTileBuffers::from_blob(per_tile_buffers_blob, n_tiles);
    PerPixelBuffers per_pixel_buffers = PerPixelBuffers::from_blob(per_pixel_buffers_blob, n_pixels, K);

    std::variant<PerInstanceBuffers<ushort>, PerInstanceBuffers<uint>> buffer_variant;
    if (end_bit <= 16) buffer_variant = PerInstanceBuffers<ushort>::from_blob(per_instance_buffers_blob, n_instances, end_bit);
    else buffer_variant = PerInstanceBuffers<uint>::from_blob(per_instance_buffers_blob, n_instances, end_bit);

    std::visit([&](auto& per_instance_buffers) {
        per_instance_buffers.primitive_indices.selector = instance_primitive_indices_selector;

        const auto blend = use_aaa_mode ? select_blend_k_kernel<true>(K) : select_blend_k_kernel<false>(K);
        blend<<<div_round_up(n_fragments, config::block_size_blend_backward), config::block_size_blend_backward>>>(
            per_tile_buffers.instance_ranges,
            per_instance_buffers.primitive_indices.Current(),
            scales,
            rotations,
            use_aaa_mode ? per_primitive_buffers.opacity : opacities,
            per_primitive_buffers.VPMT1,
            per_primitive_buffers.VPMT2,
            per_primitive_buffers.VPMT4,
            per_primitive_buffers.MT3,
            per_pixel_buffers.primitive_indices_core,
            per_pixel_buffers.grad_info_core,
            grad_image,
            grad_alpha,
            alpha,
            grad_positions,
            grad_scales,
            grad_rotations,
            grad_opacities,
            grad_sh_0,
            densification_info_helper,
            width,
            n_fragments,
            n_pixels,
            max_focal
        );
        CHECK_CUDA(config::debug_backward, "blend_backward")

    }, buffer_variant);

    const auto preprocess = use_aaa_mode ? kernels::backward::preprocess_cu<true> : kernels::backward::preprocess_cu<false>;
    preprocess<<<div_round_up(n_primitives, config::block_size_preprocess), config::block_size_preprocess>>>(
        positions,
        scales,
        rotations,
        opacities,
        sh_rest,
        per_primitive_buffers.n_touched_tiles,
        per_primitive_buffers.rgb_clamp_info,
        densification_info_helper,
        grad_positions,
        grad_opacities,
        grad_scales,
        grad_rotations,
        grad_sh_0,
        grad_sh_rest,
        densification_info,
        n_primitives,
        active_sh_bases,
        total_sh_bases,
        max_focal,
        use_distance_scaling
    );
    CHECK_CUDA(config::debug_backward, "preprocess_backward")

}
