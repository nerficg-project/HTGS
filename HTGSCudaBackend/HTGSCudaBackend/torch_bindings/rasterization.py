from typing import Any, NamedTuple
from enum import Enum
import torch
from torch.autograd.function import once_differentiable

from Cameras.Perspective import PerspectiveCamera
from Cameras.utils import IdentityDistortion, invert_camera_matrix
from Logging import Logger

from HTGSCudaBackend import _C, _C_benchmarking


class RasterizerMode(Enum):
    HYBRID_BLEND = 0
    ALPHA_BLEND_FIRST_K = 1
    ALPHA_BLEND_GLOBAL_ORDERING = 2
    OIT_BLEND = 3


class RasterizerSettings(NamedTuple):
    M: torch.Tensor  # affine transformation from model/world space to camera/view space
    VPM: torch.Tensor  # homogeneous transformation from model/world space to screen space
    cam_position: torch.Tensor  # camera position in world space
    mode: RasterizerMode
    K: int  # only used for HYBRID_BLEND and ALPHA_BLEND_FIRST_K
    active_sh_bases: int  # number of spherical harmonics bases to use for color computation
    width: int
    height: int
    near_plane: float
    far_plane: float
    scale_modifier: float  # scaling factor to be applied to each Gaussian

    def as_tuple(self) -> tuple:
        return (
            self.M,
            self.VPM,
            self.cam_position,
            self.mode.value,
            self.K,
            self.active_sh_bases,
            self.width,
            self.height,
            self.near_plane,
            self.far_plane,
            self.scale_modifier,
        )


class _Rasterize(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx: Any,
            positions: torch.Tensor,
            scales: torch.Tensor,
            rotations: torch.Tensor,
            opacities: torch.Tensor,
            sh_0: torch.Tensor,
            sh_rest: torch.Tensor,
            densification_info: torch.Tensor,
            rasterizer_settings: RasterizerSettings,
            use_distance_scaling: bool,
    ) -> torch.Tensor:
        image, per_primitive_buffers, per_tile_buffers, per_instance_buffers, per_pixel_buffers, n_instances, instance_primitive_indices_selector = _C.forward(
            positions,
            scales,
            rotations,
            opacities,
            sh_0,
            sh_rest,
            *rasterizer_settings.as_tuple(),
        )
        ctx.rasterizer_settings = rasterizer_settings
        ctx.n_instances = n_instances
        ctx.instance_primitive_indices_selector = instance_primitive_indices_selector
        ctx.save_for_backward(
            image if rasterizer_settings.mode == RasterizerMode.ALPHA_BLEND_GLOBAL_ORDERING else torch.empty(0),
            positions,
            scales,
            rotations,
            opacities if rasterizer_settings.mode == RasterizerMode.ALPHA_BLEND_FIRST_K else torch.empty(0),
            sh_rest,
            per_primitive_buffers,
            per_tile_buffers,
            per_instance_buffers,
            per_pixel_buffers,
        )
        ctx.densification_info = densification_info
        ctx.mark_non_differentiable(densification_info)
        ctx.use_distance_scaling = use_distance_scaling
        return image

    @staticmethod
    @once_differentiable
    def backward(
            ctx: Any,
            grad_image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None, None]:
        grad_positions, grad_scales, grad_rotations, grad_opacities, grad_sh_0, grad_sh_rest = _C.backward(
            ctx.densification_info,
            grad_image,
            *ctx.saved_tensors,
            *ctx.rasterizer_settings.as_tuple(),
            ctx.n_instances,
            ctx.instance_primitive_indices_selector,
            ctx.use_distance_scaling,
        )
        return (
            grad_positions,
            grad_scales,
            grad_rotations,
            grad_opacities,
            grad_sh_0,
            grad_sh_rest,
            None,  # densification_info
            None,  # rasterizer_settings
            None,  # use_distance_scaling
        )


class HTGSRasterizer(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @staticmethod
    def extract_settings(
            camera: PerspectiveCamera,
            mode: int,
            K: int,
            active_sh_bases: int,
            scale_modifier: float,
    ) -> RasterizerSettings:
        if not isinstance(camera.properties.distortion_parameters, IdentityDistortion):
            Logger.logWarning('rasterizer ignores all distortion parameters')
        try:
            rasterizer_mode = RasterizerMode(mode)
        except ValueError:
            Logger.logWarning(f'Invalid rasterizer mode: {mode}. Defaulting to HYBRID_BLEND.')
            rasterizer_mode = RasterizerMode.HYBRID_BLEND
        M = invert_camera_matrix(camera.properties.c2w)
        # flip z-axis to match other 3dgs implementations' cuda backends
        M[2] *= -1
        center_x = camera.properties.width * 0.5 + camera.properties.principal_offset_x
        center_y = camera.properties.height * 0.5 + camera.properties.principal_offset_y
        z_range = camera.far_plane - camera.near_plane
        VP = torch.tensor([
            [camera.properties.focal_x, 0.0, center_x - 0.5, 0.0],
            [0.0, camera.properties.focal_y, center_y - 0.5, 0.0],
            [0.0, 0.0, (camera.far_plane + camera.near_plane) / z_range, -2.0 * camera.far_plane * camera.near_plane / z_range],
            [0.0, 0.0, 1.0, 0.0]
        ], dtype=torch.float32, device=M.device)
        return RasterizerSettings(
            M=M,
            VPM=VP @ M,
            cam_position=camera.properties.T,
            mode=rasterizer_mode,
            K=K,
            active_sh_bases=active_sh_bases,
            width=camera.properties.width,
            height=camera.properties.height,
            near_plane=camera.near_plane,
            far_plane=camera.far_plane,
            scale_modifier=scale_modifier,
        )

    def forward(
            self,
            positions: torch.Tensor,
            scales: torch.Tensor,
            rotations: torch.Tensor,
            opacities: torch.Tensor,
            sh_0: torch.Tensor,
            sh_rest: torch.Tensor,
            densification_info: torch.Tensor,
            camera: PerspectiveCamera,
            mode: int,
            K: int,
            active_sh_bases: int,
            scale_modifier: float,
            use_distance_scaling: bool,
    ) -> torch.Tensor:
        return _Rasterize.apply(
            positions,
            scales,
            rotations,
            opacities,
            sh_0,
            sh_rest,
            densification_info,
            self.extract_settings(
                camera=camera,
                mode=mode,
                K=K,
                active_sh_bases=active_sh_bases,
                scale_modifier=scale_modifier
            ),
            use_distance_scaling,
        )

    def render(
            self,
            positions: torch.Tensor,
            scales: torch.Tensor,
            rotations: torch.Tensor,
            opacities: torch.Tensor,
            sh_0: torch.Tensor,
            sh_rest: torch.Tensor,
            camera: PerspectiveCamera,
            mode: int,
            K: int,
            active_sh_bases: int,
            scale_modifier: float,
            to_chw: bool,
            use_median_depth: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image, depth = _C.render(
            positions,
            scales,
            rotations,
            opacities,
            sh_0,
            sh_rest,
            *self.extract_settings(
                camera=camera,
                mode=mode,
                K=K,
                active_sh_bases=active_sh_bases,
                scale_modifier=scale_modifier
            ).as_tuple(),
            to_chw,
            use_median_depth,
        )
        depth = depth.unsqueeze(0) if to_chw else depth.unsqueeze(-1)
        return image, depth

    def update_max_weights(
            self,
            max_weights: torch.Tensor,
            positions: torch.Tensor,
            scales: torch.Tensor,
            rotations: torch.Tensor,
            opacities: torch.Tensor,
            camera: PerspectiveCamera,
            mode: int,
            K: int,
            active_sh_bases: int,
            scale_modifier: float,
            weight_threshold: float,
    ) -> None:
        return _C.update_max_weights(
            max_weights,
            positions,
            scales,
            rotations,
            opacities,
            *self.extract_settings(
                camera=camera,
                mode=mode,
                K=K,
                active_sh_bases=active_sh_bases,
                scale_modifier=scale_modifier
            ).as_tuple(),
            weight_threshold,
        )

    def benchmark(
            self,
            positions: torch.Tensor,
            scales: torch.Tensor,
            rotations: torch.Tensor,
            opacities: torch.Tensor,
            sh_0: torch.Tensor,
            sh_rest: torch.Tensor,
            camera: PerspectiveCamera,
            mode: int,
            K: int,
            active_sh_bases: int,
            scale_modifier: float,
            to_chw: bool,
    ) -> torch.Tensor:
        image = _C_benchmarking.benchmark(
            positions,
            scales,
            rotations,
            opacities,
            sh_0,
            sh_rest,
            *self.extract_settings(
                camera=camera,
                mode=mode,
                K=K,
                active_sh_bases=active_sh_bases,
                scale_modifier=scale_modifier
            ).as_tuple(),
            to_chw,
        )
        return image
