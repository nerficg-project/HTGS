# Efficient Perspective-Correct 3D Gaussian Splatting Using Hybrid Transparency
Florian Hahlbohm, Fabian Friederichs, Tim Weyrich, Linus Franke, Moritz Kappel, Susana Castillo, Marc Stamminger, Martin Eisemann, Marcus Magnor<br>
| [Project page](https://fhahlbohm.github.io/htgs/) | [Paper](https://arxiv.org/abs/2410.08129) | [Evaluation Images (9 GB)](https://graphics.tu-bs.de/upload/publications/hahlbohm2025htgs/htgs_full_eval.zip) | [Colab](https://colab.research.google.com/drive/1DxnIqrZ-eSSvfjhK9P1JdibABm_AJEFp?usp=sharing) |<br>

## Overview
This repository contains the official implementation of "Efficient Perspective-Correct 3D Gaussian Splatting Using Hybrid Transparency".
It is provided as an extension to the [NeRFICG](https://github.com/nerficg-project) framework.

Beyond just trying out our method, you have come to the right place if you are looking for one of the following:

- An exact and numerically stable method for computing tight screen-space bounds of a 3D Gaussian under perspective projection.
- An efficient and numerically stable approach for evaluating a 3D Gaussian at its point of maximum contribution along a ray.
- A fast and view-consistent hybrid transparency approach for blending that accelerates training and rendering without sacrificing quality.
- A fast implementation of the above, including optimized CUDA kernels.

We further provide optimized implementations for three additional blending modes alongside our default hybrid transparency blending:
1. `HYBRID_BLEND`: The K foremost fragments (core) in each pixel are alpha-blended. Remaining fragments are accumulated into an order-independent tail. Core and tail are then alpha composited to obtain the final color.
2. `ALPHA_BLEND_FIRST_K`: The K foremost fragments (core) in each pixel are alpha-blended. Remaining fragments are discarded. Same as `HYBRID_BLEND`, but without the tail.
3. `ALPHA_BLEND_GLOBAL_ORDERING`: Gaussians are "globally" sorted based on their means' z-coordinate in camera space. All fragments in each pixel are then alpha-blended in this approximate order.
4. `OIT_BLEND`: An order-independent transparency (OIT) approach that accumulates all fragments in each pixel using a weighted sum. Same as the tail in `HYBRID_BLEND`, but for all fragments.

You can find additional notes at the bottom of this page.

## Getting Started

### Our Setup
All our tests as well as experiments for the paper were conducted using the following setup:
- Operating System: Ubuntu 22.04
- GPU: Nvidia GeForce RTX 4090
- CUDA Driver Version: 535.183.01
- CUDA Toolkit Version: 11.8
- Python Version: 3.11
- PyTorch Version: 2.5.1

We have verified that everything works with CUDA Toolkit Version 12.4, but did not measure performance.
We also observed a significant performance regression (~20%) when using CUDA Driver Version 560 but are unsure of the exact reason (remaining setup was unchanged, except for the CUDA Toolkit Version where we tried both 11.8 and 12.4).

### Setup

As a preparatory step, the [NeRFICG framework](https://github.com/nerficg-project/nerficg) needs to be set up.

<details>
<summary><span style="font-weight: bold;">TL;DR NeRFICG Setup</span></summary>

- Clone the NeRFICG repository and its submodules:
	```shell
	git clone git@github.com:nerficg-project/nerficg.git --recursive && cd nerficg
	```
 
- Install the dependencies listed in `scripts/condaEnv.sh`, or automatically create a new conda environment by executing the script:
	```shell
	./scripts/condaEnv.sh && conda activate nerficg
	```
 
- [optional] For logging via [Weights & Biases](https://wandb.ai/site), run the following command and enter your account identifier:
	```shell
	wandb login
	```
 
</details>
<br>

Now, you can directly add this project as an additional method:

- Clone this repository into the `src/Methods/` directory:
	```shell
	git clone git@github.com:nerficg-project/HTGS.git src/Methods/HTGS
	```
- install all method-specific dependencies and CUDA extensions using:
	```shell
	./scripts/install.py -m HTGS
	```

## Training and Inference

The HTGS method is fully compatible with the NeRFICG scripts in the `scripts/` directory.
This includes config file generation via `defaultConfig.py`,
training via `train.py`,
inference and performance benchmarking via `inference.py`,
metric calculation via `generateTables.py`,
and live rendering via `gui.py` (Linux only).
We also used these scripts for the experiments in our paper.

For detailed instructions, please refer to the [NeRFICG framework repository](https://github.com/nerficg-project/nerficg).

### Example Configuration Files

We provide exemplary configuration files for the garden scene from the [Mip-NeRF360](https://jonbarron.info/mipnerf360/) dataset as well as the playground scene from the [Tanks and Temples](https://www.tanksandtemples.org/) dataset.
For the eight *intermediate* scenes from the Tanks and Temples dataset on which we evaluate our method in the paper, we used [our own calibration](https://cloud.tu-braunschweig.de/s/J5xYLLEdMnRwYPc) obtained using COLMAP.
We recommend copying the exemplary configuration files to the `configs/` directory.

*Note:* There will be no documentation for the method-specific configuration parameters under `TRAINING.XXXX`/`MODEL.XXXX`/`RENDERER.XXXX`.
Please conduct the code and/or our paper for understanding what they do.

### Using Custom Data

While this method is compatible with most of the dataset loaders provided with the [NeRFICG framework](https://github.com/nerficg-project/nerficg),
we recommend using exclusively the Mip-NeRF360 loader (`src/Datasets/MipNeRF360.py`) for custom data.
It is compatible with the COLMAP format for single-camera captures:
```
custom_scene
└───images
│   │   00000.jpg
│   │   ...
│   
└───sparse/0
│   │   cameras.bin
│   │   images.bin
│   │   points3D.bin
│
└───images_2  (optional)
│   │   00000.jpg
│   │   ...
```

To use it, just modify `DATASET.PATH` near the bottom of one of the exemplary configuration files. Furthermore, you may want to modify the following dataset configuration parameters:
- `DATASET.IMAGE_SCALE_FACTOR`: Set this to `null` for using the original resolution or a between zero and one to train on downscaled images.
 If `DATASET.USE_PRECOMPUTED_DOWNSCALING` is set to `true` specifying `0.5`/`0.25`/`0.125` will load images from directories `images_2`/`images_4`/`images_8` respectively.
 We recommend using this feature and downscaling manually via, e.g., `mogrify -resize 50% *.jpg` for the best results.
- `DATASET.TO_DEVICE`: Set this to `false` for large datasets or if you have less than 24 GB of VRAM.
- `DATASET.BACKGROUND_COLOR`: Will be ignored (see section "Additional Notes" for more information).
- `DATASET.NEAR_PLANE`: Must not be too small to avoid precision issues. We used `0.2` for all scenes.
- `DATASET.FAR_PLANE`: Set this generously, i.e., not too tight for your scene to avoid precision issues. We used `1000.0` for all scenes.
- `DATASET.TEST_STEP`: Set to `8` for the established evaluation protocol. Set to `0` to use all images for training.
- `DATASET.APPLY_PCA`: Tries to align the world space so that the up-axis is parallel to the direction of gravity using principal component analysis. 
 Although it does not always work, we recommend setting this to `true` if you want to view the final model inside a GUI.
 While we recommend setting `DATASET.APPLY_PCA_RESCALE` to `false`, it can be turned on to scale the scene so that all camera poses are inside the \[-1, 1\] cube.

If using your custom data fails, you have two options:
1. (Easy) Re-calibrate using, e.g., `./scripts/colmap.py -i <path/to/your/scene> --camera_mode single` and add `-u` at the end if your images are distorted.
2. (Advanced) Check the NeRFICG instructions for using custom data [here](https://github.com/nerficg-project/nerficg?tab=readme-ov-file#training-on-custom-image-sequences) and optionally dive into the NeRFICG code to extend one of the dataloaders to handle your data.

### Exporting as .ply

*Disclaimer:* The downstream application for which you use these .ply files must do a ray-based evaluation of 3D Gaussians to get the correct results.
Expect to see artifacts if the application uses the EWA splatting approach as in standard 3DGS.

We provide a script `export_ply.py` inside this repository, which extracts all 3D Gaussians from a trained model into a .ply file.
For compatibility reasons, we provide the output in the same format as the 3DGS implementation by Inria.

To use the script, move it to the `scripts/` directory of your NeRFICG installation.
Running it is similar to the `inference.py` script:
```
./scripts/export_ply.py -d output/HTGS/<OUTPUT_DIRECTORY>
```

## Additional Notes

The primary goal of this codebase is to provide a foundation for future research.
As such, we have made an effort to keep the CUDA code of the four different blending modes mostly independent of each other.
This results in a noteworthy amount of code duplication, but should allow for easy modification and extension of the individual blending modes.

We also highlight areas where we think our method could be improved or extended:

<details>
<summary><span>Densification</span></summary>

A side effect of using a ray-based evaluation for the 3D Gaussians during rendering is that the positional gradients which standard 3DGS uses for densification have significantly different properties.
Similar to much of the concurrent work in this direction, we observed this to be a major challenge and had to come up with a solution.
You can find a detailed description of our modifications in our paper.
However, we would like to clarify that the current densification strategy of our method is far from being optimal.
For example, on the bicycle scene from the Mip-NeRF360 dataset the number of Gaussians increases up to 8M in the first half of training, but then the importance pruning reduces this to 4.5M in iteration 16,000 while quality metrics go up.
Observations like these lead us to believe that an optimal densification strategy could drastically decrease training times and likely also improve reconstruction quality.

</details>

<details>
<summary><span>Anti-aliasing</span></summary>

We use a single ray through the center of each pixel for the ray-based evaluation of the 3D Gaussians.
Therefore, it is possible for Gaussians to fall between pixels making them invisible during rendering.
This is a standard aliasing problem, which the EWA splatting algorithm used in 3DGS resolves by applying a low-pass filter to the 2D covariance matrix (`+0.3`).
With a ray-based evaluation, however, this solution is not available as Gaussians are not evaluated on the image plane but in 3D space instead.
In contrast to other recent works that do ray-based 3D Gaussian rendering, we are not required to limit the minimum size of Gaussians because our inversion-free method for computing screen-space bounding boxes and ray-based evaluation can handle even degenerate Gaussians where a Gaussian's extent in one of its major axes is zero.
Nonetheless, our approach still has the aforementioned aliasing problems.
The corresponding artifacts become visible when you open a reconstructed model inside our GUI and zoom out until you see subtle flickering upon camera movement.
To avoid this being a problem during training, we employ the 3D filter from Mip-Splatting that tries to prevent 3D Gaussians from becoming smaller than a pixel in the closest training camera by applying a bit of signal theory.
We think, that this solution is far from optimal and should be addressed in the future.
It is worth noting, that a solution for this problem could likely be applied to all methods that do ray-based evaluation of 3D Gaussians or constant density ellipsoids.

</details>

<details>
<summary><span>Near plane clipping</span></summary>

Our approach for computing tight and perspective-correct bounding boxes for each 3D Gaussian is currently unable to handle certain edge-cases.
Looking at the 3D Gaussians in camera space, our approach can deal with arbitrary extents along the x-axis and y-axis, but fails to do so for certain cases with respect to extents along the z-axis.
In our implementation, we therefore cull all Gaussians for which the ellipsoid obtained by applying the used cutoff value to the 3D Gaussian is not fully between the z=near and z=far planes.
It is easy to see that this results in some Gaussians being culled, although they should be partially visible.
Especially at the near plane, this can make a major difference. 
It straightforward to extend our bounding box computation and culling to not discard Gaussians whose corresponding ellipsoid is fully between the z=0 and z=far planes instead.
However, including such Gaussians would still not be enough and further complicates how the point of maximum contribution should be calculated during blending as the point of maximum contribution along a viewing ray might not lie behind the near plane anymore.
It would be nice to see a more elegant solution that matches what is possible with, e.g., an OptiX-based implementation that uses a spatial acceleration structures to determine intersections.

</details>

<details>
<summary><span>Separate alpha thresholds for core and tail</span></summary>

To obtain good results, our hybrid transparency blending currently requires using a higher alpha threshold than what is used in standard 3DGS (`0.05` vs. `0.0039`).
While it is possible to also use `0.05` as the threshold for the tail, we found that using `0.0039` for the tail results in better quality.
We think a higher alpha threshold is needed for the core because of its limited capacity.
More precisely, we observe accurate results if the core uses most of the transmittance in each pixel, which is not possible if the core is occupied by low-alpha fragments.
Extending the blending function to account for overlap between Gaussians may resolve this issue.
An interesting byproduct of this two-threshold approach is that some view-dependent effects are represented by very large Gaussians with low opacity that will always be part of the tail.
However, further analysis is needed to understand if this is generally bad or could even be beneficial in some cases.

</details>

<details>
<summary><span>Background color blending</span></summary>

As we mainly looked at unbounded real-world scenes, input images did not have an alpha mask. This led us to use black as the background color for all scenes.
For black background, blending the background color into the final image is mathematically equivalent to not doing anything in that regard.
Therefore, our rasterization module currently only supports having a black background.
However, it should be reasonably simple to extend our rasterization module to handle arbitrary background colors during both optimization and inference.

</details>

## License and Citation
This project is licensed under the MIT license (see [LICENSE](LICENSE)).

If you use this code for your research projects, please consider a citation:
```bibtex
@article{hahlbohm2025htgs,
  title = {Efficient Perspective-Correct 3D Gaussian Splatting Using Hybrid Transparency}, 
  author = {Hahlbohm, Florian and Friederichs, Fabian and Weyrich, Tim and Franke, Linus and Kappel, Moritz and Castillo, Susana and Stamminger, Marc and Eisemann, Martin and Magnor, Marcus},
  journal = {Computer Graphics Forum},
  volume = {44},
  number = {2},
  doi = {10.1111/cgf.70014},
  year = {2025},
  url = {https://fhahlbohm.github.io/htgs/}
}
```
