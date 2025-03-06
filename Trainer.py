# -- coding: utf-8 --

"""HTGS/Trainer.py: Implementation of the trainer for HTGS."""

import torch

import Framework
from Datasets.Base import BaseDataset
from Datasets.utils import BasicPointCloud
from Logging import Logger
from Methods.Base.GuiTrainer import GuiTrainer
from Methods.Base.utils import preTrainingCallback, trainingCallback, postTrainingCallback
from Methods.HTGS.Loss import HTGSLoss
from Optim.Samplers.DatasetSamplers import DatasetSampler


@Framework.Configurable.configure(
    NUM_ITERATIONS=30_000,
    LEARNING_RATE_POSITION_INIT=0.00016,
    LEARNING_RATE_POSITION_FINAL=0.0000016,
    LEARNING_RATE_POSITION_DELAY_MULT=0.01,
    LEARNING_RATE_POSITION_MAX_STEPS=30_000,
    LEARNING_RATE_FEATURE=0.0025,
    LEARNING_RATE_OPACITY=0.05,
    LEARNING_RATE_SCALING=0.005,
    LEARNING_RATE_ROTATION=0.001,
    PERCENT_DENSE=0.01,
    USE_3D_FILTER=True,
    USE_OPACITY_RESET=False,
    OPACITY_RESET_MAX_OPACITY=0.01,
    USE_OPACITY_DECAY=True,
    USE_VISIBILITY_PRUNING=True,
    VISIBILITY_PRUNING_THRESHOLD=0.01,
    USE_DISTANCE_SCALING=True,
    OPACITY_RESET_INTERVAL=3_000,
    OPACITY_THRESHOLD=0.005,
    DENSIFY_START_ITERATION=500,
    DENSIFY_END_ITERATION=15_000,
    DENSIFICATION_INTERVAL=100,
    DENSIFY_GRAD_THRESHOLD=0.0002,
    LOSS=Framework.ConfigParameterList(
        LAMBDA_L1=0.8,
        LAMBDA_DSSIM=0.2,
    ),
)
class HTGSTrainer(GuiTrainer):
    """Defines the trainer for the HTGS method."""

    def __init__(self, **kwargs) -> None:
        super(HTGSTrainer, self).__init__(**kwargs)
        self.train_sampler = None
        self.loss = HTGSLoss(loss_config=self.LOSS)

    @preTrainingCallback(priority=50)
    @torch.no_grad()
    def createSampler(self, _, dataset: 'BaseDataset') -> None:
        """Creates the sampler."""
        self.train_sampler = DatasetSampler(dataset=dataset.train(), random=True)

    @preTrainingCallback(priority=40)
    @torch.no_grad()
    def setupGaussians(self, _, dataset: 'BaseDataset') -> None:
        """Sets up the model."""
        dataset.train()
        camera_centers = torch.stack([camera_properties.T for camera_properties in dataset])
        radius = (1.1 * torch.max(torch.linalg.norm(camera_centers - torch.mean(camera_centers, dim=0), dim=1))).item()
        Logger.logInfo(f'Training cameras extent: {radius:.2f}')

        if dataset.point_cloud is not None:
            point_cloud = dataset.point_cloud
        else:
            n_random_points = 100_000
            min_bounds, max_bounds = dataset.getBoundingBox()
            extent = max_bounds - min_bounds
            point_cloud = BasicPointCloud(torch.rand(n_random_points, 3, dtype=torch.float32, device=min_bounds.device) * extent + min_bounds)
        self.model.gaussians.initialize_from_point_cloud(point_cloud, radius)
        self.model.gaussians.training_setup(self, dataset)

    @trainingCallback(priority=110, start_iteration=1000, iteration_stride=1000)
    @torch.no_grad()
    def increaseSHDegree(self, *_) -> None:
        """Increase the number of used SH coefficients up to a maximum degree."""
        self.model.gaussians.increase_used_sh_degree()

    @trainingCallback(active='USE_VISIBILITY_PRUNING', priority=105, start_iteration=15000, iteration_stride=1000)
    @torch.no_grad()
    def importanceBasedPruning(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Pruning from RadSplat (see https://arxiv.org/abs/2403.13806)."""
        if iteration in [16000, 24000]:
            max_blending_weights = self.renderer.computeMaxWeights(dataset.train(), threshold=self.VISIBILITY_PRUNING_THRESHOLD)
            self.model.gaussians.importance_pruning(max_blending_weights, threshold=self.VISIBILITY_PRUNING_THRESHOLD)

    @trainingCallback(priority=100)
    def trainingIteration(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Performs a training step without actually doing the optimizer step."""
        # init modes
        self.model.train()
        dataset.train()
        self.loss.train()
        # update learning rate
        self.model.gaussians.update_learning_rate(iteration + 1)
        # get random sample from dataset
        camera_properties = self.train_sampler.get(dataset=dataset)['camera_properties']
        dataset.camera.setProperties(camera_properties)
        # render sample
        image = self.renderer.renderImageTraining(
            camera=dataset.camera,
            update_densification_info=iteration <= self.DENSIFY_END_ITERATION,
            use_distance_scaling=self.USE_DISTANCE_SCALING,
        )
        # calculate loss
        loss = self.loss(image, camera_properties.rgb)
        loss.backward()

    @trainingCallback(priority=90, start_iteration='DENSIFY_START_ITERATION', end_iteration='DENSIFY_END_ITERATION', iteration_stride='DENSIFICATION_INTERVAL')
    @torch.no_grad()
    def densify(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Apply densification."""
        if iteration == self.DENSIFY_START_ITERATION:
            return
        self.model.gaussians.densify_and_prune(self.DENSIFY_GRAD_THRESHOLD, self.OPACITY_THRESHOLD, iteration > self.OPACITY_RESET_INTERVAL)

        if self.USE_3D_FILTER:
            self.model.gaussians.compute_3d_filter(dataset.train())

    @trainingCallback(active='USE_OPACITY_RESET', priority=80, start_iteration='OPACITY_RESET_INTERVAL', end_iteration='DENSIFY_END_ITERATION', iteration_stride='OPACITY_RESET_INTERVAL')
    @torch.no_grad()
    def resetOpacities(self, iteration: int, _) -> None:
        """Reset opacities."""
        if iteration == self.DENSIFY_END_ITERATION:
            return
        self.model.gaussians.reset_opacities(max_opacity=self.OPACITY_RESET_MAX_OPACITY)

    @trainingCallback(active='USE_OPACITY_DECAY', priority=80, start_iteration='DENSIFY_START_ITERATION', end_iteration='DENSIFY_END_ITERATION', iteration_stride=50)
    @torch.no_grad()
    def decayOpacities(self, iteration: int, _) -> None:
        """Decay opacities."""
        if iteration == self.DENSIFY_START_ITERATION:
            return
        self.model.gaussians.decay_opacities(decay_factor=0.9995)

    @trainingCallback(active='USE_3D_FILTER', priority=75, start_iteration='DENSIFY_END_ITERATION', iteration_stride=100)
    @torch.no_grad()
    def recompute3DFilter(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Recompute 3D filter."""
        if self.DENSIFY_END_ITERATION < iteration < self.NUM_ITERATIONS - 100:
            self.model.gaussians.compute_3d_filter(dataset.train())

    @trainingCallback(priority=70)
    @torch.no_grad()
    def performOptimizerStep(self, *_) -> None:
        """Update parameters."""
        self.model.gaussians.optimizer.step()
        self.model.gaussians.optimizer.zero_grad()

    @trainingCallback(active='WANDB.ACTIVATE', priority=10, iteration_stride='WANDB.INTERVAL')
    @torch.no_grad()
    def logWandB(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Adds primitive count to default Weights & Biases logging."""
        Framework.wandb.log({
            'n_primitives': self.model.gaussians.get_positions.shape[0]
        }, step=iteration)
        # default logging
        super().logWandB(iteration, dataset)

    @postTrainingCallback(priority=1000)
    @torch.no_grad()
    def bakeActivations(self, *_) -> None:
        """Bake relevant activation functions after training."""
        self.model.gaussians.bake_activations()
        # delete optimizer to save memory
        self.model.gaussians.optimizer = None
