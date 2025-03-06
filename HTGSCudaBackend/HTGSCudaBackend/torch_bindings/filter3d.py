import torch

from Cameras.Perspective import PerspectiveCamera
from Cameras.utils import IdentityDistortion, invert_camera_matrix
from Logging import Logger

from HTGSCudaBackend import _C

def update_3d_filter(
        camera: PerspectiveCamera,
        positions: torch.Tensor,
        filter_3d: torch.Tensor,
        visibility_mask: torch.Tensor,
        clipping_tolerance: float,
        distance2filter: float,
) -> None:
    if not isinstance(camera.properties.distortion_parameters, IdentityDistortion):
        Logger.logWarning('update3dfilter ignores all distortion parameters')
    w2c = invert_camera_matrix(camera.properties.c2w.cuda())
    # flip z-axis to match other 3dgs implementations' cuda backends
    w2c[2] *= -1
    return _C.update_3d_filter_cuda(
        positions,
        w2c,
        filter_3d,
        visibility_mask,
        camera.properties.width,
        camera.properties.height,
        camera.properties.focal_x,
        camera.properties.focal_y,
        camera.properties.principal_offset_x,
        camera.properties.principal_offset_y,
        camera.near_plane,
        clipping_tolerance,
        distance2filter,
    )
