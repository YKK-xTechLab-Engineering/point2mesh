"""
Point2Mesh example scripts.

This module contains example scripts for:
- Running Point2Mesh reconstruction on various models
- Downloading example data
- Preprocessing meshes (sampling, convex hull generation)

Examples:
    Download data and run guitar reconstruction::

        from point2mesh.examples import download_data
        from point2mesh.examples.run_reconstruction import EXAMPLES, run_reconstruction

        # Download example data
        download_data()

        # Run guitar reconstruction
        run_reconstruction(EXAMPLES["guitar"])

    Sample points from a mesh::

        from point2mesh.examples import sample_mesh
        sample_mesh("input.obj", "output.ply", num_samples=75000)

Available reconstruction examples:
    - guitar: Standard guitar model
    - bull: Bull model with pooling
    - giraffe: Giraffe model
    - tiki: Tiki statue model
    - triceratops: Triceratops model
    - noisy_guitar: Guitar with noisy input (robustness demo)
    - letter_g: Letter G with beamgap loss (thin structures)
"""

# Eager imports (no pytorch3d dependency)
from .download_data import download_data

# Lazy imports for pytorch3d-dependent modules
_lazy_imports = {
    "sample_mesh": ".sample_mesh",
    "create_convex_hull": ".create_convex_hull",
    "ReconstructionConfig": ".run_reconstruction",
    "run_reconstruction": ".run_reconstruction",
    "EXAMPLES": ".run_reconstruction",
}


def __getattr__(name):
    if name in _lazy_imports:
        import importlib
        module = importlib.import_module(_lazy_imports[name], __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Data utilities
    "download_data",
    "sample_mesh",
    "create_convex_hull",
    # Reconstruction
    "ReconstructionConfig",
    "run_reconstruction",
    "EXAMPLES",
]
