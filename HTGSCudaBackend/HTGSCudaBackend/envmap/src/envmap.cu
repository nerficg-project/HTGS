#include "envmap.h"
#include "envmap_config.h"
#include "helper_math.h"
#include "utils.h"

namespace htgs::envmap {

    #define DEF inline constexpr float
    // degree 0
    DEF C0 = 0.28209479177387814f;
    // degree 1
    DEF C1 = 0.48860251190291987f;
    // degree 2
    DEF C2a = 1.0925484305920792f;
    DEF C2b = 0.94617469575755997f;
    DEF C2c = 0.31539156525251999f;
    DEF C2d = 0.54627421529603959f;
    DEF C2e = 1.8923493915151202f;
    // degree 3
    DEF C3a = 0.59004358992664352f;
    DEF C3b = 1.7701307697799304f;
    DEF C3c = 2.8906114426405538f;
    DEF C3d = 0.45704579946446572f;
    DEF C3e = 2.2852289973223288f;
    DEF C3f = 1.865881662950577f;
    DEF C3g = 1.1195289977703462f;
    DEF C3h = 1.4453057213202769f;
    DEF C3i = 3.5402615395598609f;
    DEF C3j = 4.5704579946446566f;
    DEF C3k = 5.597644988851731f;
    #undef DEF

    __global__ void forward_cu(
        const float3* __restrict__ sh_coefficients,
        const float3* __restrict__ cam_rotation,
        float* __restrict__ image,
        const int n_pixels,
        const int active_sh_bases,
        const int width,
        const int height,
        const float focal_x,
        const float focal_y,
        const float center_x,
        const float center_y,
        const bool to_chw)
    {
        const int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (pixel_idx >= n_pixels) return;

        // computation adapted from https://github.com/nerficg-project/faster-gaussian-splatting/blob/main/FasterGSCudaBackend/FasterGSCudaBackend/rasterization/include/sh_utils.cuh
        float3 result = C0 * sh_coefficients[0];
        if (active_sh_bases > 1) {
            const float local_x = (static_cast<float>(pixel_idx % width) + 0.5f - center_x) / focal_x;
            const float local_y = (static_cast<float>(pixel_idx / width) + 0.5f - center_y) / focal_y;
            const float3 cam_rotation_0 = cam_rotation[0];
            const float3 cam_rotation_1 = cam_rotation[1];
            const float3 cam_rotation_2 = cam_rotation[2];
            auto [x, y, z] = normalize(make_float3(
                cam_rotation_0.x * local_x + cam_rotation_0.y * local_y + cam_rotation_0.z,
                cam_rotation_1.x * local_x + cam_rotation_1.y * local_y + cam_rotation_1.z,
                cam_rotation_2.x * local_x + cam_rotation_2.y * local_y + cam_rotation_2.z
            ));
            result = result - C1 * y * sh_coefficients[1]
                            + C1 * z * sh_coefficients[2]
                            - C1 * x * sh_coefficients[3];
            if (active_sh_bases > 4) {
                const float xx = x * x, yy = y * y, zz = z * z;
                const float xy = x * y, xz = x * z, yz = y * z;
                result = result + C2a * xy * sh_coefficients[4]
                                - C2a * yz * sh_coefficients[5]
                                + (C2b * zz - C2c) * sh_coefficients[6]
                                - C2a * xz * sh_coefficients[7]
                                + C2d * (xx - yy) * sh_coefficients[8];
                if (active_sh_bases > 9) {
                    result = result + y * (C3a * yy - C3b * xx) * sh_coefficients[9]
                                    + C3c * xy * z * sh_coefficients[10]
                                    + y * (C3d - C3e * zz) * sh_coefficients[11]
                                    + z * (C3f * zz - C3g) * sh_coefficients[12]
                                    + x * (C3d - C3e * zz) * sh_coefficients[13]
                                    + C3h * z * (xx - yy) * sh_coefficients[14]
                                    + x * (C3b * yy - C3a * xx) * sh_coefficients[15];
                }
            }
        }

        if (to_chw) {
            image[pixel_idx] = result.x;
            image[n_pixels + pixel_idx] = result.y;
            image[2 * n_pixels + pixel_idx] = result.z;
        }
        else {
            const int base_idx = 3 * pixel_idx;
            image[base_idx] = result.x;
            image[base_idx + 1] = result.y;
            image[base_idx + 2] = result.z;
        }
    }

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
        const bool to_chw)
    {
        if constexpr (config::debug) {
            if (!sh_coefficients.is_cuda() || !sh_coefficients.is_floating_point() || !sh_coefficients.is_contiguous())
                throw std::runtime_error("Input tensor 'sh_coefficients' must be a contiguous CUDA float tensor.");
        }

        const int n_pixels = width * height;
        const torch::TensorOptions float_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
        torch::Tensor image = to_chw ? torch::empty({3, height, width}, float_options) : torch::empty({height, width, 3}, float_options);

        forward_cu<<<div_round_up(n_pixels, config::block_size), config::block_size>>>(
            reinterpret_cast<float3*>(sh_coefficients.data_ptr<float>()),
            reinterpret_cast<float3*>(cam_rotation.contiguous().data_ptr<float>()),
            image.data_ptr<float>(),
            n_pixels,
            active_sh_bases,
            width,
            height,
            focal_x,
            focal_y,
            center_x,
            center_y,
            to_chw
        );
        CHECK_CUDA(config::debug, "forward")

        return image;
    }

    __global__ void backward_cu(
        const float* __restrict__ grad_image,
        const float3* __restrict__ cam_rotation,
        float3* __restrict__ grad_helper,
        const int n_pixels,
        const int active_sh_bases,
        const int width,
        const int height,
        const float focal_x,
        const float focal_y,
        const float center_x,
        const float center_y,
        const bool to_chw)
    {
        const int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (pixel_idx >= n_pixels) return;

        float3 grad_pixel;
        if (to_chw) {
            grad_pixel.x = grad_image[pixel_idx];
            grad_pixel.y = grad_image[n_pixels + pixel_idx];
            grad_pixel.z = grad_image[2 * n_pixels + pixel_idx];
        }
        else {
            const int base_idx = 3 * pixel_idx;
            grad_pixel.x = grad_image[base_idx];
            grad_pixel.y = grad_image[base_idx + 1];
            grad_pixel.z = grad_image[base_idx + 2];
        }

        // computation adapted from https://github.com/nerficg-project/faster-gaussian-splatting/blob/main/FasterGSCudaBackend/FasterGSCudaBackend/rasterization/include/sh_utils.cuh
        grad_helper[pixel_idx] = C0 * grad_pixel;
        if (active_sh_bases > 1) {
            const float local_x = (static_cast<float>(pixel_idx % width) + 0.5f - center_x) / focal_x;
            const float local_y = (static_cast<float>(pixel_idx / width) + 0.5f - center_y) / focal_y;
            const float3 cam_rotation_0 = cam_rotation[0];
            const float3 cam_rotation_1 = cam_rotation[1];
            const float3 cam_rotation_2 = cam_rotation[2];
            auto [x, y, z] = normalize(make_float3(
                cam_rotation_0.x * local_x + cam_rotation_0.y * local_y + cam_rotation_0.z,
                cam_rotation_1.x * local_x + cam_rotation_1.y * local_y + cam_rotation_1.z,
                cam_rotation_2.x * local_x + cam_rotation_2.y * local_y + cam_rotation_2.z
            ));
            grad_helper[n_pixels + pixel_idx] = -C1 * y * grad_pixel;
            grad_helper[2 * n_pixels + pixel_idx] = C1 * z * grad_pixel;
            grad_helper[3 * n_pixels + pixel_idx] = -C1 * x * grad_pixel;
            if (active_sh_bases > 4) {
                const float xx = x * x, yy = y * y, zz = z * z;
                const float xy = x * y, xz = x * z, yz = y * z;
                grad_helper[4 * n_pixels + pixel_idx] = C2a * xy * grad_pixel;
                grad_helper[5 * n_pixels + pixel_idx] = -C2a * yz * grad_pixel;
                grad_helper[6 * n_pixels + pixel_idx] = (C2b * zz - C2c) * grad_pixel;
                grad_helper[7 * n_pixels + pixel_idx] = -C2a * xz * grad_pixel;
                grad_helper[8 * n_pixels + pixel_idx] = C2d * (xx - yy) * grad_pixel;
                if (active_sh_bases > 9) {
                    grad_helper[9 * n_pixels + pixel_idx] = y * (C3a * yy - C3b * xx) * grad_pixel;
                    grad_helper[10 * n_pixels + pixel_idx] = C3c * xy * z * grad_pixel;
                    grad_helper[11 * n_pixels + pixel_idx] = y * (C3d - C3e * zz) * grad_pixel;
                    grad_helper[12 * n_pixels + pixel_idx] = z * (C3f * zz - C3g) * grad_pixel;
                    grad_helper[13 * n_pixels + pixel_idx] = x * (C3d - C3e * zz) * grad_pixel;
                    grad_helper[14 * n_pixels + pixel_idx] = C3h * z * (xx - yy) * grad_pixel;
                    grad_helper[15 * n_pixels + pixel_idx] = x * (C3b * yy - C3a * xx) * grad_pixel;
                }
            }
        }
    }

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
        const bool to_chw)
    {
        const int n_pixels = width * height;
        const torch::TensorOptions float_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
        torch::Tensor grad_helper = torch::empty({active_sh_bases, n_pixels, 3}, float_options);

        backward_cu<<<div_round_up(n_pixels, config::block_size), config::block_size>>>(
            grad_image.contiguous().data_ptr<float>(),
            reinterpret_cast<float3*>(cam_rotation.contiguous().data_ptr<float>()),
            reinterpret_cast<float3*>(grad_helper.data_ptr<float>()),
            n_pixels,
            active_sh_bases,
            width,
            height,
            focal_x,
            focal_y,
            center_x,
            center_y,
            to_chw
        );
        CHECK_CUDA(config::debug, "backward")

        torch::Tensor grad_sh_coefficients = torch::zeros({total_sh_bases, 3}, float_options);
        grad_sh_coefficients.index_put_({torch::indexing::Slice(0, active_sh_bases)}, grad_helper.sum(1));

        return grad_sh_coefficients;
    }

}