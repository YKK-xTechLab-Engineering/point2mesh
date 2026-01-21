"""
Point2Mesh: A Self-Prior for Deformable Meshes

This package implements Point2Mesh, a technique for reconstructing a surface mesh
from an input point cloud by optimizing the weights of a CNN.

Reference:
    Hanocka et al., "Point2Mesh: A Self-Prior for Deformable Meshes", SIGGRAPH 2020
    https://github.com/ranahanocka/point2mesh
"""

__version__ = "0.1.0"

# Eager imports (no pytorch3d dependency)
from .options import Options
from .utils import (
    read_pts,
    load_obj,
    export,
    manifold_upsample,
    get_manifold_bin_dir,
    get_manifold_executable,
    get_simplify_executable,
)

# Lazy imports for pytorch3d-dependent modules
_lazy_imports = {
    "Mesh": ".models.layers.mesh",
    "PartMesh": ".models.layers.mesh",
    "PriorNet": ".models.networks",
    "PartNet": ".models.networks",
    "MeshEncoderDecoder": ".models.networks",
    "init_net": ".models.networks",
    "sample_surface": ".models.networks",
    "local_nonuniform_penalty": ".models.networks",
    "chamfer_distance": ".models.losses",
    "BeamGapLoss": ".models.losses",
}


def __getattr__(name):
    if name in _lazy_imports:
        import importlib
        module = importlib.import_module(_lazy_imports[name], __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Options",
    "get_manifold_bin_dir",
    "get_manifold_executable",
    "get_simplify_executable",
    "read_pts",
    "load_obj",
    "export",
    "manifold_upsample",
    "Mesh",
    "PartMesh",
    "PriorNet",
    "PartNet",
    "MeshEncoderDecoder",
    "init_net",
    "sample_surface",
    "local_nonuniform_penalty",
    "chamfer_distance",
    "BeamGapLoss",
]
