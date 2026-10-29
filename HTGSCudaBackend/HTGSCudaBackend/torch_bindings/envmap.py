from typing import Any, NamedTuple

import torch
from torch.autograd.function import once_differentiable

from HTGSCudaBackend import _C


class EnvmapSettings(NamedTuple):
    cam_rotation: torch.Tensor
    total_sh_bases: int
    active_sh_bases: int
    width: int
    height: int
    focal_x: float
    focal_y: float
    center_x: float
    center_y: float
    to_chw: bool

    def as_tuple(self) -> tuple:
        return (
            self.cam_rotation,
            self.total_sh_bases,
            self.active_sh_bases,
            self.width,
            self.height,
            self.focal_x,
            self.focal_y,
            self.center_x,
            self.center_y,
            self.to_chw,
        )


class _Envmap(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        sh_coefficients: torch.Tensor,
        envmap_settings: EnvmapSettings,
    ) -> torch.Tensor:
        image = _C.envmap_forward(
            sh_coefficients,
            *envmap_settings.as_tuple(),
        )
        ctx.envmap_settings = envmap_settings
        return image

    @staticmethod
    @once_differentiable
    def backward(
        ctx: Any,
        grad_image: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        grad_sh_coefficients = _C.envmap_backward(
            grad_image,
            *ctx.envmap_settings.as_tuple(),
        )
        return (
            grad_sh_coefficients,
            None,  # envmap_settings
        )


def eval_envmap(
    sh_coefficients: torch.Tensor,
    cam_rotation: torch.Tensor,
    active_sh_bases: int,
    width: int,
    height: int,
    focal_x: float,
    focal_y: float,
    center_x: float,
    center_y: float,
    to_chw: bool,
) -> torch.Tensor:
    envmap_settings = EnvmapSettings(
        cam_rotation,
        sh_coefficients.shape[0],
        active_sh_bases,
        width,
        height,
        focal_x,
        focal_y,
        center_x,
        center_y,
        to_chw,
    )
    return _Envmap.apply(
        sh_coefficients,
        envmap_settings,
    )
