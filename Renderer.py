"""HTGS/Renderer.py: Defines the renderer for the HTGS method."""

import torch
import numpy as np

import Framework
from Cameras.Perspective import PerspectiveCamera
from Datasets.Base import BaseDataset
from Datasets.utils import View
from Logging import Logger
from Methods.HTGS.HTGSCudaBackend import diff_rasterize, rasterize, fast_rasterize, update_max_weights, RasterizerSettings, RasterizerMode
from Methods.Base.Renderer import BaseModel
from Methods.Base.Renderer import BaseRenderer
from Methods.HTGS.Model import HTGSModel
from Visual.utils import apply_color_map


def extract_settings(
    view: View,
    mode: int,
    K: int,
    active_sh_bases: int,
    scale_modifier: float,
) -> RasterizerSettings:
    if not isinstance(view.camera, PerspectiveCamera):
        raise Framework.RendererError('HTGS renderer only supports perspective cameras')
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
        near_plane=near_plane,
        far_plane=far_plane,
        scale_modifier=scale_modifier,
    )


@Framework.Configurable.configure(
    BLEND_MODE=0,
    K=16,
    SCALE_MODIFIER=1.0,
    DISABLE_SH0=False,
    DISABLE_SH1=False,
    DISABLE_SH2=False,
    DISABLE_SH3=False,
    USE_MEDIAN_DEPTH=False,
    FORCE_OPTIMIZED_INFERENCE=False,
)
class HTGSRenderer(BaseRenderer):
    """Renderer for the HTGS method."""

    def __init__(self, model: 'BaseModel') -> None:
        super().__init__(model, [HTGSModel])
        if not Framework.config.GLOBAL.GPU_INDICES:
            raise Framework.RendererError('renderer not implemented in CPU mode')
        if len(Framework.config.GLOBAL.GPU_INDICES) > 1:
            Logger.log_warning(f'renderer not implemented in multi-GPU mode: using GPU {Framework.config.GLOBAL.GPU_INDICES[0]}')
        if self.BLEND_MODE not in [0, 1, 2, 3]:
            raise Framework.RendererError('Invalid blend mode')
        if self.BLEND_MODE < 2 and self.K not in [1, 2, 4, 8, 16, 32]:
            Logger.log_warning(f'unsupported K value for selected blend mode may lead to undefined behavior: {self.K}')

    def render_image(self, view: View, to_chw: bool = False, benchmark: bool = False) -> dict[str, torch.Tensor]:
        """Renders an image for a given view."""
        if benchmark or self.FORCE_OPTIMIZED_INFERENCE:
            return self.render_image_benchmark(view, to_chw=to_chw or benchmark)
        elif self.model.training:
            raise Framework.RendererError('please directly call render_image_training() instead of render_image() during training')
        else:
            return self.render_image_inference(view, to_chw)

    def render_image_training(self, view: View, update_densification_info: bool, use_distance_scaling: bool) -> torch.Tensor:
        """Renders an image for a given view for optimization."""
        settings = extract_settings(
            view=view,
            mode=self.BLEND_MODE,
            K=self.K,
            active_sh_bases=self.model.gaussians.active_sh_bases,
            scale_modifier=1.0,
        )
        return diff_rasterize(
            positions=self.model.gaussians.get_positions,
            scales=self.model.gaussians.get_scales_with_3D_filter if self.model.gaussians.use_3d_filter else self.model.gaussians.get_scales,
            rotations=self.model.gaussians.get_rotations,
            opacities=self.model.gaussians.get_opacities_with_3D_filter if self.model.gaussians.use_3d_filter else self.model.gaussians.get_opacities,
            sh_0=self.model.gaussians.get_sh_0,
            sh_rest=self.model.gaussians.get_sh_rest,
            densification_info=self.model.gaussians.get_densification_info if update_densification_info else torch.empty(0),
            settings=settings,
            use_distance_scaling=use_distance_scaling and update_densification_info,
        )

    @torch.no_grad()
    def render_image_inference(self, view: View, to_chw: bool) -> dict[str, torch.Tensor]:
        """Renders an image for a given view during inference."""
        # modify sh features for visualization
        sh_0 = self.model.gaussians.get_sh_0
        if self.DISABLE_SH0:
            sh_0 = torch.zeros_like(sh_0)
        sh_rest = self.model.gaussians.get_sh_rest
        if self.DISABLE_SH1 or self.DISABLE_SH2 or self.DISABLE_SH3:
            sh_rest = sh_rest.clone()
        if self.DISABLE_SH1:
            sh_rest[:, 0:3].zero_()
        if self.DISABLE_SH2:
            sh_rest[:, 3:8].zero_()
        if self.DISABLE_SH3:
            sh_rest[:, 8:15].zero_()

        settings = extract_settings(
            view=view,
            mode=self.BLEND_MODE,
            K=self.K,
            active_sh_bases=self.model.gaussians.active_sh_bases,
            scale_modifier=self.SCALE_MODIFIER,
        )
        rgb, depth = rasterize(
            positions=self.model.gaussians.get_positions,
            scales=self.model.gaussians.get_scales_with_3D_filter if self.model.gaussians.use_3d_filter else self.model.gaussians.get_scales,
            rotations=self.model.gaussians.get_rotations,
            opacities=self.model.gaussians.get_opacities_with_3D_filter if self.model.gaussians.use_3d_filter else self.model.gaussians.get_opacities,
            sh_0=sh_0,
            sh_rest=sh_rest,
            settings=settings,
            to_chw=to_chw,
            use_median_depth=self.USE_MEDIAN_DEPTH,
        )
        return {
            'rgb': rgb,
            'depth': depth
        }

    @torch.inference_mode()
    def render_image_benchmark(self, view: View, to_chw: bool) -> dict[str, torch.Tensor]:
        """Renders a only the rgb image "as fast as possible"."""
        settings = extract_settings(
            view=view,
            mode=self.BLEND_MODE,
            K=self.K,
            active_sh_bases=self.model.gaussians.active_sh_bases,
            scale_modifier=self.SCALE_MODIFIER,
        )
        rgb = fast_rasterize(
            positions=self.model.gaussians.get_positions,
            scales=self.model.gaussians.get_scales_with_3D_filter if self.model.gaussians.use_3d_filter else self.model.gaussians.get_scales,
            rotations=self.model.gaussians.get_rotations,
            opacities=self.model.gaussians.get_opacities_with_3D_filter if self.model.gaussians.use_3d_filter else self.model.gaussians.get_opacities,
            sh_0=self.model.gaussians.get_sh_0,
            sh_rest=self.model.gaussians.get_sh_rest,
            settings=settings,
            to_chw=to_chw,
        )
        return { 'rgb': rgb }

    def compute_max_weights(self, dataset: BaseDataset, threshold: float) -> torch.Tensor:
        """Computes the maximum blending weights for the current dataset."""
        positions = self.model.gaussians.get_positions
        scales = self.model.gaussians.get_scales_with_3D_filter if self.model.gaussians.use_3d_filter else self.model.gaussians.get_scales
        opacities = self.model.gaussians.get_opacities_with_3D_filter if self.model.gaussians.use_3d_filter else self.model.gaussians.get_opacities
        rotations = self.model.gaussians.get_rotations
        max_weights = torch.zeros(opacities.shape[0], device=opacities.device, dtype=opacities.dtype)
        for view in dataset:
            settings = extract_settings(
                view=view,
                mode=self.BLEND_MODE,
                K=self.K,
                active_sh_bases=0,
                scale_modifier=1.0,
            )
            update_max_weights(
                max_weights=max_weights,
                positions=positions,
                scales=scales,
                rotations=rotations,
                opacities=opacities,
                settings=settings,
                weight_threshold=threshold,
            )
        return max_weights

    def postprocess_outputs(self, outputs: dict[str, torch.Tensor], *_) -> dict[str, torch.Tensor]:
        """Postprocesses the model outputs, returning tensors of shape 3xHxW."""
        return {
            'rgb': outputs['rgb'],
            'depth': apply_color_map(
                color_map='SPECTRAL',
                image=outputs['depth'],
                min_max=None,
                mask=outputs['depth'] > 0.0,
                interpolate=True
            ),
        }
