"""HTGS/Model.py: Implementation of the model for the HTGS method."""

import torch
import numpy as np

import Framework
from Cameras.Perspective import PerspectiveCamera
from Datasets.Base import BaseDataset
from Datasets.utils import BasicPointCloud, View
from Cameras.utils import quaternion_to_rotation_matrix, directions_to_equirectangular_grid_coords, \
    equirectangular_grid_coords_to_directions
from Logging import Logger
from Methods.Base.Model import BaseModel
from Optim.adam_utils import replace_param_group_data, prune_param_groups, extend_param_groups
from Optim.lr_utils import LRDecayPolicy
from Optim.knn_utils import compute_root_mean_squared_knn_distances
import Thirdparty.TinyCudaNN as tcnn
from CudaUtils.MortonEncoding import morton_encode
from Methods.HTGS.HTGSCudaBackend import eval_envmap, update_3d_filter
from Methods.HTGS.utils import compute_discard_mask


class NeuralEnvironmentMap(torch.nn.Module):
    """Environment map using spherical harmonics encoding and an MLP."""

    def __init__(self, width: int = 2000, height: int = 1000) -> None:
        """Initialize submodules. Width and height are relevant only for baking."""
        super().__init__()
        self.width = width
        self.height = height
        self._baked_texture = None
        self.net_with_encoding = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=3,
            encoding_config={
                'otype': 'SphericalHarmonics',
                'degree': 4
            },
            network_config={
                'otype': 'FullyFusedMLP',
                'activation': 'ReLU',
                'output_activation': 'Sigmoid',
                'n_neurons': 128,
                # 'n_hidden_layers': 4,
                'n_hidden_layers': 2,
            },
            seed=Framework.config.GLOBAL.RANDOM_SEED
        )

    def train(self, mode: bool = True) -> 'NeuralEnvironmentMap':
        """Switches the module's mode, invalidating the baked texture as the weights may change."""
        self._baked_texture = None
        return super().train(mode)

    def get_optimizer_param_groups(self, max_iterations: int) -> tuple[list[dict], list[LRDecayPolicy]]:
        """Returns the parameter groups for the optimizer."""
        param_groups = [{'params': self.net_with_encoding.parameters(), 'lr': 1.0}]
        schedulers = [self.lr_decay_policy]
        return param_groups, schedulers

    def training_setup(self, training_wrapper) -> None:
        self.lr_decay_policy = LRDecayPolicy(
            lr_init=training_wrapper.LEARNING_RATE_BACKGROUND_INIT * 0.1,
            lr_final=training_wrapper.LEARNING_RATE_BACKGROUND_FINAL * 0.1,
            lr_delay_steps=0,
            lr_delay_mult=1.0,
            max_steps=training_wrapper.LEARNING_RATE_BACKGROUND_MAX_STEPS,
        )

    def increase_used_sh_degree(self) -> None:
        """Increases the used SH degree."""
        pass

    @property
    @torch.no_grad()
    def get_baked_texture(self) -> torch.Tensor:
        """Returns the MLP baked into an equirectangular texture, evaluated at the texel centers sampled by
        EnvironmentMap. The values are post-activation, as the output activation is part of the network.
        The result is cached until the module's mode is switched."""
        if self._baked_texture is None:
            # texel centers in grid coordinates, i.e. the align_corners=False convention used for sampling
            u = torch.arange(self.width, dtype=torch.float32).add_(0.5).mul_(2.0 / self.width).sub_(1.0)
            v = torch.arange(self.height, dtype=torch.float32).add_(0.5).mul_(2.0 / self.height).sub_(1.0)
            grid_v, grid_u = torch.meshgrid(v, u, indexing='ij')
            directions = equirectangular_grid_coords_to_directions(torch.stack([grid_u, grid_v], dim=-1).view(-1, 2))
            directions.mul_(0.5).add_(0.5)  # for compatibility with the SH encoding in tiny-cuda-nn
            self._baked_texture = self.net_with_encoding(directions).float().view(self.height, self.width, 3)\
                .permute(2, 0, 1).contiguous()
        return self._baked_texture

    def forward(self, view: View, to_chw: bool) -> torch.Tensor:
        """Returns environment map values for the given view."""
        view_dirs_all = view.cam_to_world(view.camera.compute_local_ray_directions(), is_point=False)
        view_dirs_all = torch.nn.functional.normalize(view_dirs_all, p=2, dim=-1)
        view_dirs_all.mul_(0.5).add_(0.5)  # for compatibility with the SH encoding in tiny-cuda-nn
        image = self.net_with_encoding(view_dirs_all).reshape(view.camera.height, view.camera.width, 3)
        return image.permute(2, 0, 1) if to_chw else image


class SHEnvironmentMap(torch.nn.Module):
    """Environment map using spherical harmonics encoding. Uses fused CUDA evaluation for SH degrees <= 3, otherwise falls back to TCNN."""

    def __init__(self, sh_degree: int, pretrained: bool = False, disable_coarse_to_fine: bool = False) -> None:
        """Initialize submodules."""
        super().__init__()
        self.active_sh_degree = sh_degree if pretrained or disable_coarse_to_fine else 0
        self.active_sh_bases = (self.active_sh_degree + 1) ** 2
        self.max_sh_degree = sh_degree
        if sh_degree > 3:
            self.encoding = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    'otype': 'SphericalHarmonics',
                    'degree': sh_degree + 1,  # +1 because tcnn counts degrees differently
                },
                dtype=torch.float32,
                seed=Framework.config.GLOBAL.RANDOM_SEED
            )
            self.register_parameter('coefficients', torch.nn.Parameter(torch.zeros((1, self.encoding.n_output_dims, 3), dtype=torch.float32)))
        else:
            self.encoding = None
            self.register_parameter('coefficients', torch.nn.Parameter(torch.zeros((1, (sh_degree + 1) ** 2, 3), dtype=torch.float32)))

        # TODO
        #   - compare different output activations
        #   - learning rate tuning
        #   - fused implementation
        #   - binary cross entropy loss tuning
        # self.output_activation = torch.nn.Sigmoid()
        # self.output_activation = lambda x: x.add(0.5).clamp(min=0.0, max=1.0)
        self.output_activation = lambda x: torch.tanh(x) * 0.5 + 0.5

    def training_setup(self, training_wrapper) -> None:
        self.lr_decay_policy = LRDecayPolicy(
            lr_init=training_wrapper.LEARNING_RATE_BACKGROUND_INIT,
            lr_final=training_wrapper.LEARNING_RATE_BACKGROUND_FINAL,
            lr_delay_steps=0,
            lr_delay_mult=1.0,
            max_steps=training_wrapper.LEARNING_RATE_BACKGROUND_MAX_STEPS,
        )

    def get_optimizer_param_groups(self, max_iterations: int) -> tuple[list[dict], list[LRDecayPolicy]]:
        """Returns the parameter groups for the optimizer."""
        param_groups = [{'params': self.parameters(), 'lr': 1.0}]
        schedulers = [self.lr_decay_policy]
        return param_groups, schedulers

    def increase_used_sh_degree(self) -> None:
        """Increases the used SH degree."""
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
            self.active_sh_bases = (self.active_sh_degree + 1) ** 2

    @property
    def get_sh_coefficients(self) -> torch.Tensor:
        """Returns all SH coefficients as a (bases, 3) tensor. Coefficients of inactive degrees are zero."""
        return self.coefficients[0]

    def eval_sh(self, view_directions: torch.Tensor) -> torch.Tensor:
        """Evaluates the SH coefficients for the given view directions."""
        sh_values = self.encoding(view_directions)[:, :self.active_sh_bases, None]
        coefficients = self.coefficients[:, :self.active_sh_bases]
        return self.output_activation(sh_values.mul(coefficients).sum(dim=-2))

    def forward(self, view: View, to_chw: bool) -> torch.Tensor:
        """Returns environment map values for the given view."""
        if self.max_sh_degree <= 3 and isinstance(view.camera, PerspectiveCamera):
            return self.output_activation(eval_envmap(
                sh_coefficients=self.coefficients.squeeze(0),
                cam_rotation=view.rotation,
                active_sh_bases=self.active_sh_bases,
                width=view.camera.width,
                height=view.camera.height,
                focal_x=view.camera.focal_x,
                focal_y=view.camera.focal_y,
                center_x=view.camera.center_x,
                center_y=view.camera.center_y,
                to_chw=to_chw,
            ))

        view_dirs_all = view.cam_to_world(view.camera.compute_local_ray_directions(), is_point=False)
        view_dirs_all = torch.nn.functional.normalize(view_dirs_all, p=2, dim=-1)
        view_dirs_all.mul_(0.5).add_(0.5)  # for compatibility with the SH encoding in tiny-cuda-nn
        image = self.eval_sh(view_dirs_all).reshape(view.camera.height, view.camera.width, 3)
        return image.permute(2, 0, 1) if to_chw else image


class EnvironmentMap(torch.nn.Module):
    """Environment map."""

    def __init__(self, width: int, height: int) -> None:
        """Initialize submodules."""
        super().__init__()
        self.register_parameter('texture', torch.nn.Parameter(
            torch.full((1, 3, height, width), fill_value=0.5, dtype=torch.float32)
        ))

    def training_setup(self, training_wrapper) -> None:
        self.lr_decay_policy = LRDecayPolicy(
            lr_init=training_wrapper.LEARNING_RATE_BACKGROUND_INIT * 0.1,
            lr_final=training_wrapper.LEARNING_RATE_BACKGROUND_FINAL * 0.3,
            lr_delay_steps=500,
            lr_delay_mult=0.1,
            max_steps=training_wrapper.LEARNING_RATE_BACKGROUND_MAX_STEPS,
        )

    def get_optimizer_param_groups(self, max_iterations: int) -> tuple[list[dict], list[LRDecayPolicy]]:
        """Returns the parameter groups for the optimizer."""
        param_groups = [{'params': self.parameters(), 'lr': 1.0}]
        schedulers = [self.lr_decay_policy]
        return param_groups, schedulers

    def increase_used_sh_degree(self) -> None:
        """Increases the used SH degree."""
        pass

    @property
    def get_texture(self) -> torch.Tensor:
        """Returns the raw texture, i.e. without the output activation applied."""
        return self.texture[0]

    @property
    def get_baked_texture(self) -> torch.Tensor:
        """Returns the pre-activated texture. Note that interpolating the activated texture != activating the
        interpolated texture, so training behavior is not preserved!"""
        return self.texture[0].sigmoid()

    def forward(self, view: View, to_chw: bool) -> torch.Tensor:
        """Returns environment map values for the given view."""
        view_directions = view.cam_to_world(view.camera.compute_local_ray_directions(), is_point=False)
        view_directions = torch.nn.functional.normalize(view_directions, p=2, dim=-1)
        grid_coords = directions_to_equirectangular_grid_coords(view_directions)
        image = torch.nn.functional.grid_sample(
            input=self.texture,  # (1, 3, height, width)
            grid=grid_coords.view(1, view.camera.height, view.camera.width, 2),
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        ).squeeze().sigmoid()  # (3, height, width)
        return image if to_chw else image.permute(1, 2, 0)


class Gaussians(torch.nn.Module):
    """Stores a set of points with 3D Gaussian extent."""

    GOF_DENSIFICATION_GRAD = True
    GOF_DENSIFICATION_CLONE = True

    def __init__(self, sh_degree: int, pretrained: bool) -> None:
        super().__init__()
        self.active_sh_degree = sh_degree if pretrained else 0
        self.active_sh_bases = (self.active_sh_degree + 1) ** 2
        self.max_sh_degree = sh_degree
        self.register_parameter('_positions', None)
        self.register_parameter('_sh_0', None)
        self.register_parameter('_sh_rest', None)
        self.register_parameter('_scales', None)
        self.register_parameter('_rotations', None)
        self.register_parameter('_opacities', None)
        self.densification_info = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.training_cameras_extent = 1.0
        self.filter_3D = torch.empty(0)
        self.use_3d_filter = False
        self.distance2filter = 0
        # activation functions
        self.scale_activation = torch.nn.Identity() if pretrained else torch.exp
        self.inverse_scale_activation = torch.nn.Identity() if pretrained else torch.log
        self.opacity_activation = torch.nn.Identity() if pretrained else torch.special.expit
        self.inverse_opacity_activation = torch.nn.Identity() if pretrained else torch.special.logit
        self.rotation_activation = torch.nn.Identity() if pretrained else torch.nn.functional.normalize

    @property
    def get_scales(self) -> torch.Tensor:
        """Returns the Gaussians' scales."""
        return self.scale_activation(self._scales)

    @property
    def get_rotations(self) -> torch.Tensor:
        """Returns the Gaussians' rotations as quaternions."""
        return self.rotation_activation(self._rotations)

    @property
    def get_positions(self) -> torch.Tensor:
        """Returns the Gaussians' means."""
        return self._positions

    @property
    def get_sh_0(self) -> torch.Tensor:
        """Returns the Gaussians' 0-th degree SH features."""
        return self._sh_0

    @property
    def get_sh_rest(self) -> torch.Tensor:
        """Returns the Gaussians' SH features beyond the 0-th degree."""
        return self._sh_rest

    @property
    def get_opacities(self) -> torch.Tensor:
        """Returns the Gaussians' opacities."""
        return self.opacity_activation(self._opacities)

    @property
    def get_opacities_with_3D_filter(self) -> torch.Tensor:
        """Returns the Gaussians' opacities with the 3D filter applied."""
        # apply 3D filter
        scales = self.get_scales
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)
        scales_after_square = scales_square + torch.square(self.filter_3D)
        det2 = scales_after_square.prod(dim=1)
        coef = torch.sqrt(det1 / det2)
        return self.get_opacities * coef[..., None]

    @property
    def get_scales_with_3D_filter(self) -> torch.Tensor:
        """Returns the Gaussians' scales with the 3D filter applied."""
        scales = self.get_scales
        # apply 3D filter
        scales = torch.square(scales) + torch.square(self.filter_3D)
        scales = torch.sqrt(scales)
        return scales

    @property
    def get_densification_info(self) -> torch.Tensor:
        """Returns the current densification info buffers."""
        return self.densification_info

    def setup_3d_filter(self, dataset: 'BaseDataset', dilation: float = 0.2) -> None:
        """Sets up a 3D filter (see https://arxiv.org/abs/2311.16493)."""
        self.use_3d_filter = True
        max_focal = 1.0e-12
        for view in dataset:
            if not isinstance(view.camera, PerspectiveCamera):
                raise Framework.ModelError('3d filter only supports perspective cameras')
            if view.camera.distortion is not None:
                Logger.log_warning('3d filter ignores all distortion parameters')
            max_focal = max(max_focal, max(view.camera.focal_x, view.camera.focal_y))
        # assume max_focal is focal length of the highest resolution camera
        self.distance2filter = dilation ** 0.5 / max_focal
        self.compute_3d_filter(dataset)

    def compute_3d_filter(self, dataset: 'BaseDataset', clipping_tolerance: float = 0.15) -> None:
        """Computes the 3D filter."""
        positions = self.get_positions.contiguous()
        filter_3d = torch.full((positions.shape[0], 1), fill_value=torch.finfo(torch.float32).max, device=positions.device, dtype=torch.float32)
        visibility_mask = torch.zeros((positions.shape[0], 1), device=positions.device, dtype=torch.bool)
        for view in dataset:
            if not isinstance(view.camera, PerspectiveCamera):
                raise Framework.ModelError('3d filter only supports perspective cameras')
            if view.camera.distortion is not None:
                Logger.log_warning('3d filter ignores all distortion parameters')
            update_3d_filter(
                positions,
                view.w2c,
                filter_3d,
                visibility_mask,
                view.camera.width,
                view.camera.height,
                view.camera.focal_x,
                view.camera.focal_y,
                view.camera.center_x,
                view.camera.center_y,
                view.camera.near_plane,
                clipping_tolerance,
                self.distance2filter,
            )
        filter_3d_max = filter_3d[visibility_mask].max()
        filter_3d = torch.where(visibility_mask, filter_3d, filter_3d_max, out=filter_3d)
        self.filter_3D = filter_3d

    def increase_used_sh_degree(self) -> None:
        """Increases the used SH degree."""
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
            self.active_sh_bases = (self.active_sh_degree + 1) ** 2

    def initialize_from_point_cloud(self, point_cloud: BasicPointCloud, training_cameras_extent: float) -> None:
        """Initializes the model from a point cloud."""
        self.training_cameras_extent = training_cameras_extent
        positions = point_cloud.positions.cuda()
        rgbs = torch.full_like(positions, fill_value=0.5) if point_cloud.colors is None else point_cloud.colors.cuda()
        n_initial_points = positions.shape[0]
        sh_all = torch.zeros((n_initial_points, (self.max_sh_degree + 1) ** 2, 3), dtype=torch.float32, device='cuda')
        sh_all[:, 0] = (rgbs - 0.5) / 0.28209479177387814

        Logger.log_info(f'Number of points at initialization: {n_initial_points:,}')

        distances = compute_root_mean_squared_knn_distances(positions)
        scales = self.inverse_scale_activation(distances)[..., None].repeat(1, 3)
        rotations = torch.zeros((n_initial_points, 4), dtype=torch.float32, device='cuda')
        rotations[:, 0] = 1.0

        opacities = self.inverse_opacity_activation(torch.full((n_initial_points, 1), fill_value=0.1, dtype=torch.float32, device='cuda'))

        self._positions = torch.nn.Parameter(positions.contiguous())
        self._sh_0 = torch.nn.Parameter(sh_all[:, 0:1].contiguous())
        self._sh_rest = torch.nn.Parameter(sh_all[:, 1:].contiguous())
        self._scales = torch.nn.Parameter(scales.contiguous())
        self._rotations = torch.nn.Parameter(rotations.contiguous())
        self._opacities = torch.nn.Parameter(opacities.contiguous())
        self.reset_densification_info()

    def training_setup(self, training_wrapper) -> None:
        """Sets up the optimizer."""
        self.percent_dense = training_wrapper.PERCENT_DENSE

        param_groups = [
            {'params': [self._positions], 'lr': training_wrapper.LEARNING_RATE_POSITION_INIT * self.training_cameras_extent, 'name': 'positions'},
            {'params': [self._sh_0], 'lr': training_wrapper.LEARNING_RATE_FEATURE, 'name': 'sh_0'},
            {'params': [self._sh_rest], 'lr': training_wrapper.LEARNING_RATE_FEATURE / 20.0, 'name': 'sh_rest'},
            {'params': [self._opacities], 'lr': training_wrapper.LEARNING_RATE_OPACITY, 'name': 'opacities'},
            {'params': [self._scales], 'lr': training_wrapper.LEARNING_RATE_SCALING, 'name': 'scales'},
            {'params': [self._rotations], 'lr': training_wrapper.LEARNING_RATE_ROTATION, 'name': 'rotations'}
        ]

        try:
            from Thirdparty.Apex import FusedAdam
            # slightly faster than the PyTorch implementation
            self.optimizer = FusedAdam(param_groups, lr=0.0, eps=1e-15, adam_w_mode=False)
            Logger.log_info('using apex FusedAdam')
        except Framework.ExtensionError:
            Logger.log_warning('apex is not installed -> using the slightly slower PyTorch Adam instead')
            Logger.log_warning('apex can be installed using ./scripts/install.py -e src/Thirdparty/Apex.py')
            self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15, fused=True)

        self.position_lr_scheduler = LRDecayPolicy(
            lr_init=training_wrapper.LEARNING_RATE_POSITION_INIT * self.training_cameras_extent,
            lr_final=training_wrapper.LEARNING_RATE_POSITION_FINAL * self.training_cameras_extent,
            max_steps=training_wrapper.LEARNING_RATE_POSITION_MAX_STEPS)

    def update_learning_rate(self, iteration: int) -> None:
        """ Learning rate scheduling per step """
        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'positions':
                lr = self.position_lr_scheduler(iteration)
                param_group['lr'] = lr

    def reset_opacities(self, max_opacity: float) -> None:
        """Resets the opacities to a fixed value."""
        current_opacities = self.get_opacities_with_3D_filter if self.use_3d_filter else self.get_opacities
        opacities_new = current_opacities.clamp_max(max_opacity)
        if self.use_3d_filter:
            # make sure that the current 3d filter has the same effect on the new opacities
            scales_square = torch.square(self.get_scales)
            det1 = scales_square.prod(dim=1)
            scales_after_square = scales_square + torch.square(self.filter_3D)
            det2 = scales_after_square.prod(dim=1)
            coef = torch.sqrt(det1 / det2)
            opacities_new = opacities_new / coef[..., None]
        opacities_new = self.inverse_opacity_activation(opacities_new)
        replace_param_group_data(self.optimizer, opacities_new, 'opacities')

    def decay_opacities(self, decay_factor: float):
        """Decays the opacities by a factor."""
        opacities_new = self.inverse_opacity_activation(self.get_opacities * decay_factor)
        replace_param_group_data(self.optimizer, opacities_new, 'opacities')
        # these lines match the behavior of reset_opacities, but aren't necessary as the decay is only a small change
        # current_opacities = self.get_opacities_with_3D_filter if self.use_3d_filter else self.get_opacities
        # opacities_new = current_opacities * decay_factor
        # if self.use_3d_filter:
        #     # make sure that the current 3d filter has the same effect on the new opacities
        #     scales = self.get_scales
        #     scales_square = torch.square(scales)
        #     det1 = scales_square.prod(dim=1)
        #     scales_after_square = scales_square + torch.square(self.filter_3D)
        #     det2 = scales_after_square.prod(dim=1)
        #     coef = torch.sqrt(det1 / det2)
        #     opacities_new = opacities_new / coef[..., None]
        # opacities_new = self.inverse_opacity_activation(opacities_new)
        # replace_param_group_data(self.optimizer, opacities_new, 'opacities')

    def prune_points(self, prune_mask: torch.Tensor) -> None:
        """Prunes points that are not visible or too large."""
        valid_mask = ~prune_mask
        optimizable_tensors = prune_param_groups(self.optimizer, valid_mask)

        self._positions = optimizable_tensors['positions']
        self._sh_0 = optimizable_tensors['sh_0']
        self._sh_rest = optimizable_tensors['sh_rest']
        self._opacities = optimizable_tensors['opacities']
        self._scales = optimizable_tensors['scales']
        self._rotations = optimizable_tensors['rotations']

    def densification_postfix(
            self,
            new_positions: torch.Tensor,
            new_sh_0: torch.Tensor,
            new_sh_rest: torch.Tensor,
            new_opacities: torch.Tensor,
            new_scales: torch.Tensor,
            new_rotations: torch.Tensor
    ) -> None:
        """Incorporate the changes from the densification step into the parameter groups."""
        optimizable_tensors = extend_param_groups(self.optimizer, {
            'positions': new_positions,
            'sh_0': new_sh_0,
            'sh_rest': new_sh_rest,
            'opacities': new_opacities,
            'scales': new_scales,
            'rotations': new_rotations
        })
        self._positions = optimizable_tensors['positions']
        self._sh_0 = optimizable_tensors['sh_0']
        self._sh_rest = optimizable_tensors['sh_rest']
        self._opacities = optimizable_tensors['opacities']
        self._scales = optimizable_tensors['scales']
        self._rotations = optimizable_tensors['rotations']

    def reset_densification_info(self) -> None:
        """Resets the densification info buffers to zero."""
        n_points = self._positions.shape[0]
        n_floats = 3 if Gaussians.GOF_DENSIFICATION_GRAD else 2
        self.densification_info = torch.zeros((n_floats, n_points, 1), dtype=torch.float32, device='cuda')

    def split(self, grads: torch.Tensor, grad_threshold: float, grads_abs: torch.Tensor | None, grad_abs_threshold: float | None) -> torch.Tensor:
        """Densify by splitting Gaussians that satisfy the gradient condition."""
        n_init_points = self.get_positions.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros(n_init_points, dtype=torch.float32, device='cuda')
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = padded_grad >= grad_threshold
        if grads_abs is not None:
            padded_grad_abs = torch.zeros(n_init_points, dtype=torch.float32, device='cuda')
            padded_grad_abs[:grads_abs.shape[0]] = grads_abs.squeeze()
            selected_pts_mask |= padded_grad_abs >= grad_abs_threshold
        selected_pts_mask &= torch.max(self.get_scales, dim=1).values > self.percent_dense * self.training_cameras_extent

        stds = self.get_scales[selected_pts_mask].repeat(2, 1)
        samples = torch.normal(mean=0.0, std=stds)
        rots = quaternion_to_rotation_matrix(self._rotations[selected_pts_mask]).repeat(2, 1, 1)
        new_positions = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_positions[selected_pts_mask].repeat(2, 1)
        new_scales = self.inverse_scale_activation(self.get_scales[selected_pts_mask].repeat(2, 1) / 1.6)
        new_rotations = self._rotations[selected_pts_mask].repeat(2, 1)
        new_sh_0 = self._sh_0[selected_pts_mask].repeat(2, 1, 1)
        new_sh_rest = self._sh_rest[selected_pts_mask].repeat(2, 1, 1)
        new_opacities = self._opacities[selected_pts_mask].repeat(2, 1)

        self.densification_postfix(new_positions, new_sh_0, new_sh_rest, new_opacities, new_scales, new_rotations)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(2 * selected_pts_mask.sum().item(), device='cuda', dtype=torch.bool)))
        return prune_filter

    def duplicate(self, grads: torch.Tensor, grad_threshold: float, grads_abs: torch.Tensor | None, grad_abs_threshold: float | None) -> None:
        """Densify by duplicating Gaussians that satisfy the gradient condition."""
        # Extract points that satisfy the gradient condition
        selected_pts_mask = grads.flatten() >= grad_threshold
        if grads_abs is not None:
            selected_pts_mask |= grads_abs.flatten() >= grad_abs_threshold
        selected_pts_mask &= torch.max(self.get_scales, dim=1).values <= self.percent_dense * self.training_cameras_extent

        if Gaussians.GOF_DENSIFICATION_CLONE:
            # sample a new gaussian instead of fixing position (from gof)
            stds = self.get_scales[selected_pts_mask]
            samples = torch.normal(mean=0.0, std=stds)
            rots = quaternion_to_rotation_matrix(self._rotations[selected_pts_mask])
            new_positions = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_positions[selected_pts_mask]
        else:
            new_positions = self._positions[selected_pts_mask]  # 3dgs

        new_sh_0 = self._sh_0[selected_pts_mask]
        new_sh_rest = self._sh_rest[selected_pts_mask]
        new_opacities = self._opacities[selected_pts_mask]
        new_scales = self._scales[selected_pts_mask]
        new_rotations = self._rotations[selected_pts_mask]

        self.densification_postfix(new_positions, new_sh_0, new_sh_rest, new_opacities, new_scales, new_rotations)

    def densify_and_prune(self, grad_threshold: float, min_opacity: float, prune_large_gaussians: bool) -> None:
        """Densifies the point cloud and prunes points that are not visible or too large."""
        denominator = self.densification_info[0].clamp_min(1.0)
        grads = self.densification_info[1] / denominator
        grads_abs, grad_abs_threshold = None, None
        if Gaussians.GOF_DENSIFICATION_GRAD:
            grads_abs = self.densification_info[2] / denominator
            ratio = (grads.flatten() >= grad_threshold).float().mean()
            grad_abs_threshold = torch.quantile(grads_abs.flatten(), 1.0 - ratio).item()

        self.duplicate(grads, grad_threshold, grads_abs, grad_abs_threshold)
        prune_mask = self.split(grads, grad_threshold, grads_abs, grad_abs_threshold)

        prune_mask |= self.get_opacities.flatten() < min_opacity
        if prune_large_gaussians:
            prune_mask |= self.get_scales.max(dim=1).values > 0.1 * self.training_cameras_extent
        self.prune_points(prune_mask)

        self.reset_densification_info()

        torch.cuda.empty_cache()

    def importance_pruning(self, max_blending_weights: torch.Tensor, threshold: float) -> None:
        """Prunes points based on the maximum blending weights."""
        mask = max_blending_weights < threshold
        self.prune_points(mask)
        if self.use_3d_filter:
            self.filter_3D = self.filter_3D[~mask].contiguous()
        self.densification_info = self.densification_info[:, ~mask].contiguous()

    def bake_activations(self):
        """Bakes relevant activation functions into the final parameters."""
        # bake activation functions into final parameters
        self._rotations.data = self.get_rotations
        self.rotation_activation = torch.nn.Identity()
        # Important: opacities must be baked before scales due to implementation of get_opacities_with_3D_filter
        self._opacities.data = self.get_opacities_with_3D_filter if self.use_3d_filter else self.get_opacities
        self.opacity_activation = torch.nn.Identity()
        self.inverse_opacity_activation = torch.nn.Identity()
        self._scales.data = self.get_scales_with_3D_filter if self.use_3d_filter else self.get_scales
        self.scale_activation = torch.nn.Identity()
        self.inverse_scale_activation = torch.nn.Identity()
        # 3d filter is baked into relevant parameters now
        self.use_3d_filter = False

        # prune points that would never be visible anyway
        self.prune_points((self._opacities < 0.00392156862).squeeze())  # 1/255

        # morton sort
        morton_encoding = morton_encode(self._positions)
        order = torch.argsort(morton_encoding)
        self._positions.data = self._positions[order].contiguous()
        self._rotations.data = self._rotations[order].contiguous()
        self._sh_0.data = self._sh_0[order].contiguous()
        self._sh_rest.data = self._sh_rest[order].contiguous()
        self._scales.data = self._scales[order].contiguous()
        self._opacities.data = self._opacities[order].contiguous()

    @torch.no_grad()
    def as_ply_dict(self) -> dict[str, np.ndarray]:
        """Returns the model as a ply-compatible dictionary using structured numpy arrays."""
        if self.get_positions.shape[0] == 0:
            return {}

        # construct attributes
        positions = self.get_positions.detach().contiguous().cpu().numpy()
        sh_0 = self.get_sh_0.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        sh_rest = self.get_sh_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self.get_opacities.logit().detach().contiguous().cpu().numpy()  # most viewers expect unactivated opacities
        scales = self.get_scales.log().detach().contiguous().cpu().numpy()  # most viewers expect unactivated scales
        rotations = self.get_rotations.detach().contiguous().cpu().numpy()
        attributes = np.concatenate((positions, sh_0, sh_rest, opacities, scales, rotations), axis=1)

        # construct structured array
        attribute_names = (
              ['x', 'y', 'z']                                    # 3d mean
            + ['f_dc_0', 'f_dc_1', 'f_dc_2']                     # 0-th SH degree coefficients
            + [f'f_rest_{i}' for i in range(sh_rest.shape[-1])]  # remaining SH degree coefficients
            + ['opacity']                                        # opacity (pre-activation)
            + ['scale_0', 'scale_1', 'scale_2']                  # 3d scale (pre-activation)
            + ['rot_0', 'rot_1', 'rot_2', 'rot_3']               # rotation quaternion
        )
        dtype = 'f4'  # store all attributes as float32 for compatibility
        full_dtype = [(attribute_name, dtype) for attribute_name in attribute_names]
        vertices = np.empty(positions.shape[0], dtype=full_dtype)

        # insert attributes into structured array
        vertices[:] = list(map(tuple, attributes))

        return {'vertex': vertices}


@Framework.Configurable.configure(
    SH_DEGREE=3,
    BACKGROUND_MODEL='SH',  # 'NONE', 'SH', 'MLP', or 'TEX'
    BACKGROUND_DISCARD_BELOW=0.1,  # discard the background below this y component of the view direction, None to disable
    SH_BACKGROUND_MODEL=Framework.ConfigParameterList(
        DEGREE=3,
        DISABLE_COARSE_TO_FINE=False,
    ),
    ENVIRONMENT_MAP=Framework.ConfigParameterList(
        TEXTURE_WIDTH=2000,
        TEXTURE_HEIGHT=1000,
    ),
)
class HTGSModel(BaseModel):
    """Defines the HTGS model."""

    def __init__(self, name: str = None) -> None:
        super().__init__(name)
        self.gaussians: Gaussians | None = None
        self.background_model: SHEnvironmentMap | NeuralEnvironmentMap | EnvironmentMap | None = None

    def build(self) -> 'HTGSModel':
        """Builds the model."""
        pretrained = self.num_iterations_trained > 0
        self.gaussians = Gaussians(self.SH_DEGREE, pretrained)
        match self.BACKGROUND_MODEL:
            case 'NONE':
                self.background_model = None
            case 'SH':
                self.background_model = SHEnvironmentMap(
                    sh_degree=self.SH_BACKGROUND_MODEL.DEGREE,
                    pretrained=pretrained,
                    disable_coarse_to_fine=self.SH_BACKGROUND_MODEL.DISABLE_COARSE_TO_FINE,
                )
            case 'MLP':
                self.background_model = NeuralEnvironmentMap(
                    width=self.ENVIRONMENT_MAP.TEXTURE_WIDTH,
                    height=self.ENVIRONMENT_MAP.TEXTURE_HEIGHT,
                )
            case 'TEX':
                self.background_model = EnvironmentMap(
                    width=self.ENVIRONMENT_MAP.TEXTURE_WIDTH,
                    height=self.ENVIRONMENT_MAP.TEXTURE_HEIGHT,
                )
            case _:
                raise Framework.ModelError(f'MODEL.BACKGROUND_MODEL must be one of NONE/SH/MLP/TEX, got {self.BACKGROUND_MODEL}')
        return self

    def eval_background(self, view: View, to_chw: bool, discard_color: torch.Tensor | None = None) -> torch.Tensor:
        """Evaluates the background model for the given view, discarding outputs below the
        horizon defined by MODEL.BACKGROUND_COLOR_DISCORD_BELOW."""
        background = self.background_model(view, to_chw)
        if self.BACKGROUND_DISCARD_BELOW is None:
            return background
        discarded = compute_discard_mask(view, self.BACKGROUND_DISCARD_BELOW, to_chw)
        if discard_color is None:
            return background.masked_fill(discarded, 0.0)
        return torch.where(discarded, discard_color[:, None, None] if to_chw else discard_color, background)

    def get_optimizer_param_groups(self, n_iterations: int) -> tuple[list[dict], list[LRDecayPolicy]]:
        """Returns the optimizer parameter groups and learning rate schedulers."""
        learnable_components = [self.background_model]
        param_groups, schedulers = zip(*(component.get_optimizer_param_groups(n_iterations) for component in learnable_components))
        return sum(param_groups, []), sum(schedulers, [])

    def get_ply_dict(self) -> dict[str, np.ndarray | list[str]]:
        """Returns the model as a ply-compatible dictionary using structured numpy arrays."""
        data: dict[str, np.ndarray | list[str]] = {}
        if self.gaussians is None or not (data := self.gaussians.as_ply_dict()):
            return data

        # add method-specific comments
        data['comments'] = ['SplatRenderMode: default', 'Generated with NeRFICG/HTGS']

        return data

