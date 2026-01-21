"""Point2Mesh mesh layers."""

# Lazy imports for pytorch3d-dependent modules
_lazy_imports = {
    "Mesh": ".mesh",
    "PartMesh": ".mesh",
    "MeshConv": ".mesh_conv",
    "MeshPool": ".mesh_pool",
    "MeshUnpool": ".mesh_unpool",
    "MeshUnion": ".mesh_union",
}


def __getattr__(name):
    if name in _lazy_imports:
        import importlib
        module = importlib.import_module(_lazy_imports[name], __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Mesh",
    "PartMesh",
    "MeshConv",
    "MeshPool",
    "MeshUnpool",
    "MeshUnion",
]
