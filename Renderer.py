# -- coding: utf-8 --

"""HTGS/Renderer.py: """

import torch

import Framework
from Cameras.Perspective import PerspectiveCamera
from Datasets.Base import BaseDataset
from Logging import Logger
from Methods.HTGS.HTGSCudaBackend import HTGSRasterizer
from Methods.Base.Renderer import BaseModel
from Methods.Base.Renderer import BaseRenderer
from Methods.HTGS.Model import HTGSModel
from Visual.utils import pseudoColorDepth


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
            Logger.logWarning(f'renderer not implemented in multi-GPU mode: using GPU {Framework.config.GLOBAL.GPU_INDICES[0]}')
        self.rasterizer = HTGSRasterizer()
        if self.BLEND_MODE not in [0, 1, 2, 3]:
            raise Framework.RendererError('Invalid blend mode')
        if self.BLEND_MODE < 2 and self.K not in [1, 2, 4, 8, 16, 32]:
            Logger.logWarning(f'unsupported K value for selected blend mode may lead to undefined behavior: {self.K}')

    def renderImage(self, camera: 'PerspectiveCamera', to_chw: bool = False, benchmark: bool = False) -> dict[str, torch.Tensor]:
        """Renders an image for a given camera."""
        if benchmark or self.FORCE_OPTIMIZED_INFERENCE:
            return self.renderImageBenchmark(camera, to_chw=to_chw or benchmark)
        else:
            return self.renderImageInference(camera, to_chw)

    def renderImageTraining(self, camera: 'PerspectiveCamera', update_densification_info: bool, use_distance_scaling: bool) -> torch.Tensor:
        """Renders an image for a given camera for optimization."""
        return self.rasterizer(
            positions=self.model.gaussians.get_positions,
            scales=self.model.gaussians.get_scales_with_3D_filter if self.model.gaussians.use_3d_filter else self.model.gaussians.get_scales,
            rotations=self.model.gaussians.get_rotations,
            opacities=self.model.gaussians.get_opacities_with_3D_filter if self.model.gaussians.use_3d_filter else self.model.gaussians.get_opacities,
            sh_0=self.model.gaussians.get_sh_0,
            sh_rest=self.model.gaussians.get_sh_rest,
            densification_info=self.model.gaussians.get_densification_info if update_densification_info else torch.empty(0),
            camera=camera,
            mode=self.BLEND_MODE,
            K=self.K,
            active_sh_bases=self.model.gaussians.active_sh_bases,
            scale_modifier=1.0,
            use_distance_scaling=use_distance_scaling and update_densification_info,
        )

    @torch.no_grad()
    def renderImageInference(self, camera: 'PerspectiveCamera', to_chw: bool) -> dict[str, torch.Tensor]:
        """Renders an image for a given camera during inference."""
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

        rgb, depth = self.rasterizer.render(
            positions=self.model.gaussians.get_positions,
            scales=self.model.gaussians.get_scales_with_3D_filter if self.model.gaussians.use_3d_filter else self.model.gaussians.get_scales,
            rotations=self.model.gaussians.get_rotations,
            opacities=self.model.gaussians.get_opacities_with_3D_filter if self.model.gaussians.use_3d_filter else self.model.gaussians.get_opacities,
            sh_0=sh_0,
            sh_rest=sh_rest,
            camera=camera,
            mode=self.BLEND_MODE,
            K=self.K,
            active_sh_bases=self.model.gaussians.active_sh_bases,
            scale_modifier=self.SCALE_MODIFIER,
            to_chw=to_chw,
            use_median_depth=self.USE_MEDIAN_DEPTH,
        )
        return {
            'rgb': rgb,
            'depth': depth
        }

    @torch.inference_mode()
    def renderImageBenchmark(self, camera: 'PerspectiveCamera', to_chw: bool) -> dict[str, torch.Tensor]:
        """Renders an image for a given camera."""
        rgb = self.rasterizer.benchmark(
            positions=self.model.gaussians.get_positions,
            scales=self.model.gaussians.get_scales_with_3D_filter if self.model.gaussians.use_3d_filter else self.model.gaussians.get_scales,
            rotations=self.model.gaussians.get_rotations,
            opacities=self.model.gaussians.get_opacities_with_3D_filter if self.model.gaussians.use_3d_filter else self.model.gaussians.get_opacities,
            sh_0=self.model.gaussians.get_sh_0,
            sh_rest=self.model.gaussians.get_sh_rest,
            camera=camera,
            mode=self.BLEND_MODE,
            K=self.K,
            active_sh_bases=self.model.gaussians.active_sh_bases,
            scale_modifier=self.SCALE_MODIFIER,
            to_chw=to_chw,
        )
        return { 'rgb': rgb }

    def computeMaxWeights(self, dataset: BaseDataset, threshold: float) -> torch.Tensor:
        """Computes the maximum blending weights for the current dataset."""
        positions = self.model.gaussians.get_positions
        scales = self.model.gaussians.get_scales_with_3D_filter if self.model.gaussians.use_3d_filter else self.model.gaussians.get_scales
        opacities = self.model.gaussians.get_opacities_with_3D_filter if self.model.gaussians.use_3d_filter else self.model.gaussians.get_opacities
        rotations = self.model.gaussians.get_rotations
        max_weights = torch.zeros(opacities.shape[0], device=opacities.device, dtype=opacities.dtype)
        # we do not use the default dataset iterator to avoid copies, thus it is important to not modify anything here
        for camera_properties in dataset.data[dataset.mode]:
            dataset.camera.setProperties(camera_properties)
            self.rasterizer.update_max_weights(
                max_weights=max_weights,
                positions=positions,
                scales=scales,
                rotations=rotations,
                opacities=opacities,
                camera=dataset.camera,
                mode=self.BLEND_MODE,
                K=self.K,
                active_sh_bases=0,
                scale_modifier=1.0,
                weight_threshold=threshold,
            )
        return max_weights

    def pseudoColorOutputs(self, outputs: dict[str, torch.Tensor | None], camera: 'BaseCamera', dataset: BaseDataset, index: int) -> dict[str, torch.Tensor]:
        """Pseudo-colors the model outputs, returning tensors of shape 3xHxW."""
        return {
            'rgb': outputs['rgb'],
            'depth': pseudoColorDepth(
                color_map='SPECTRAL',
                depth=outputs['depth'],
                near_far=None,
                alpha=outputs['depth'] > 0.0,
                interpolate=True
            ),
        }

    def pseudoColorGT(self, camera: 'BaseCamera', dataset: BaseDataset, index: int) -> dict[str, torch.Tensor]:
        """Pseudo-colors the gt labels relevant for this method, returning tensors of shape 3xHxW."""
        gt_data = {}
        if camera.properties.rgb is not None:
            gt_data['rgb_gt'] = camera.properties.rgb
        return gt_data
