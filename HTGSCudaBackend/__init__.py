from pathlib import Path

import Framework

extension_dir = Path(__file__).parent
__extension_name__ = extension_dir.name
__install_command__ = [
    'pip', 'install',
    str(extension_dir),
    '--no-build-isolation',  # to build the extension using the current environment instead of creating a new one
]

try:
    from .HTGSCudaBackend.torch_bindings.rasterization import HTGSRasterizer
    from .HTGSCudaBackend.torch_bindings.filter3d import update_3d_filter
    __all__ = ['HTGSRasterizer', 'update_3d_filter']
except ImportError as e:
    raise Framework.ExtensionError(name=__extension_name__, install_command=__install_command__)
