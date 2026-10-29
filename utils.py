"""HTGS/utils.py"""

import torch
import numpy as np

import Framework
from Logging import Logger
from Datasets.utils import View
from Cameras.Perspective import PerspectiveCamera
from Methods.HTGS.HTGSCudaBackend import RasterizerSettings, RasterizerMode


def compute_discard_mask(view: View, discard_below: float, to_chw: bool) -> torch.Tensor:
    """Returns a mask that is True for the pixels below the given horizon, i.e. the ones whose normalized
    world space ray direction has a y component (pointing down) larger than discard_below."""
    local_directions = view.camera.compute_local_ray_directions()
    # the world space y component is the dot product with the second row of the camera-to-world rotation
    direction_y = local_directions.matmul(view.rotation[1]).div_(torch.linalg.norm(local_directions, dim=-1))
    mask = direction_y > discard_below
    return mask.view(1, view.camera.height, view.camera.width) if to_chw \
        else mask.view(view.camera.height, view.camera.width, 1)


def extract_settings(
    view: View,
    mode: int,
    K: int,
    active_sh_bases: int,
    scale_modifier: float,
    use_aaa_mode: bool,
) -> RasterizerSettings:
    if not isinstance(view.camera, PerspectiveCamera):
        raise Framework.RendererError('HTGSv2 renderer only supports perspective cameras')
    if view.camera.distortion is not None:
        Logger.log_warning('found distortion parameters that will be ignored by the rasterizer')
    try:
        rasterizer_mode = RasterizerMode(mode)
    except ValueError:
        Logger.log_warning(f'Invalid rasterizer mode: {mode}. Defaulting to HYBRID_BLEND.')
        rasterizer_mode = RasterizerMode.HYBRID_BLEND
    M = view.w2c_numpy
    near_plane = view.camera.near_plane
    far_plane = view.camera.far_plane
    z_range = far_plane - near_plane
    VP = np.array([
        [view.camera.focal_x, 0.0, view.camera.center_x - 0.5, 0.0],
        [0.0, view.camera.focal_y, view.camera.center_y - 0.5, 0.0],
        [0.0, 0.0, (far_plane + near_plane) / z_range, -2.0 * far_plane * near_plane / z_range],
        [0.0, 0.0, 1.0, 0.0]
    ], dtype=M.dtype)
    VPM = VP @ M
    return RasterizerSettings(
        M=torch.as_tensor(M, dtype=torch.float32, device='cuda'),
        VPM=torch.as_tensor(VPM, dtype=torch.float32, device='cuda'),
        cam_position=view.position,
        mode=rasterizer_mode,
        K=K,
        active_sh_bases=active_sh_bases,
        width=view.camera.width,
        height=view.camera.height,
        focal_x=view.camera.focal_x,
        focal_y=view.camera.focal_y,
        center_x=view.camera.center_x,
        center_y=view.camera.center_y,
        near_plane=near_plane,
        far_plane=far_plane,
        scale_modifier=scale_modifier,
        use_aaa_mode=use_aaa_mode,
    )
