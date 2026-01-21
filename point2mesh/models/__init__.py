"""Point2Mesh neural network models."""

# Lazy imports for pytorch3d-dependent modules
_lazy_imports = {
    "PriorNet": ".networks",
    "PartNet": ".networks",
    "MeshEncoderDecoder": ".networks",
    "MeshEncoder": ".networks",
    "MeshDecoder": ".networks",
    "init_net": ".networks",
    "sample_surface": ".networks",
    "local_nonuniform_penalty": ".networks",
    "chamfer_distance": ".losses",
    "BeamGapLoss": ".losses",
}


def __getattr__(name):
    if name in _lazy_imports:
        import importlib
        module = importlib.import_module(_lazy_imports[name], __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PriorNet",
    "PartNet",
    "MeshEncoderDecoder",
    "MeshEncoder",
    "MeshDecoder",
    "init_net",
    "sample_surface",
    "local_nonuniform_penalty",
    "chamfer_distance",
    "BeamGapLoss",
]
