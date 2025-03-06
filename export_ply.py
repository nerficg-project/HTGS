#! /usr/bin/env python3
# -- coding: utf-8 --

"""export_ply.py: Extracts a trained HTGS into a .ply file."""

from argparse import ArgumentParser
from pathlib import Path

import torch
from plyfile import PlyData, PlyElement
import numpy as np

import utils
with utils.discoverSourcePath():
    import Framework
    from Logging import Logger
    from Implementations import Methods as MI
    from Methods.GaussianSplatting.utils import inverse_sigmoid


def construct_list_of_attributes(n_floats_sh_rest: int):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2']
    for i in range(n_floats_sh_rest):
        l.append(f'f_rest_{i}')
    l.append('opacity')
    l.append('scale_0')
    l.append('scale_1')
    l.append('scale_2')
    l.append('rot_0')
    l.append('rot_1')
    l.append('rot_2')
    l.append('rot_3')
    return l


def save_as_ply(
        output_path: Path,
        positions: torch.Tensor,
        sh_0: torch.Tensor,
        sh_rest: torch.Tensor,
        opacities: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
) -> None:
    normals = np.zeros_like(positions)
    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(sh_rest.shape[-1])]
    elements = np.empty(positions.shape[0], dtype=dtype_full)
    attributes = np.concatenate((positions, normals, sh_0, sh_rest, opacities, scales, rotations), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(output_path)


def main(*, base_dir: Path) -> None:
    # setup framework
    Framework.setup(config_path=str(base_dir / 'training_config.yaml'), require_custom_config=True)
    # load model
    model = MI.getModel(
        method=Framework.config.GLOBAL.METHOD_TYPE,
        checkpoint=str(base_dir / 'checkpoints' / 'final.pt'),
    ).eval()
    # extract Gaussians
    positions = model.gaussians.get_positions.detach()
    positions = positions.cpu().numpy()
    sh_0 = model.gaussians.get_sh_0.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    sh_rest = model.gaussians.get_sh_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = inverse_sigmoid(model.gaussians.get_opacities).detach().cpu().numpy()
    scales = torch.log(model.gaussians.get_scales).detach().cpu().numpy()
    rotations = model.gaussians.get_rotations.detach()
    rotations = rotations.cpu().numpy()

    Logger.logInfo('starting ply export')
    # create and write .ply file
    save_as_ply(
        base_dir / 'gaussians.ply',
        positions,
        sh_0,
        sh_rest,
        opacities,
        scales,
        rotations,
    )
    Logger.logInfo('done')


if __name__ == '__main__':
    parser = ArgumentParser(prog='Export_PLY')
    parser.add_argument(
        '-d', '--dir', action='store', dest='base_dir', default=None,
        metavar='path/to/output/directory', required=True,
        help='A directory containing the outputs of a completed training.'
    )
    args, _ = parser.parse_known_args()
    Logger.setMode(Logger.MODE_VERBOSE)
    main(base_dir=Path(args.base_dir))
