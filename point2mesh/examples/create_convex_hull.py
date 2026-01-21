"""
Create a convex hull initial mesh from a point cloud.

This script generates an initial convex hull mesh from a point cloud,
which can be used as the starting mesh for Point2Mesh reconstruction.
"""

import argparse
import os
import subprocess
import warnings
from pathlib import Path
from typing import Optional

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

from ..utils import read_pts, export


def count_faces(path: Path) -> int:
    """Count the number of faces in an OBJ file."""
    with open(path, "r") as f:
        lines = f.read().split("\n")
    return sum(1 for line in lines if line.startswith("f"))


def run_manifold(path: Path, res: int, manifold_path: Path) -> None:
    """Run manifold software to make mesh watertight."""
    cmd = f"{manifold_path}/manifold {path} {path} {res}"
    os.system(cmd)


def run_simplify(path: Path, faces: int, manifold_path: Path) -> None:
    """Run simplify software to reduce face count."""
    cmd = f"{manifold_path}/simplify -i {path} -o {path} -f {faces}"
    os.system(cmd)


def run_blender_hull(
    target: Path, dest: Path, res: int, blender_path: Path
) -> None:
    """Run Blender to generate convex hull."""
    script_path = Path(__file__).parent / "blender_scripts" / "blender_hull.py"
    cmd = (
        f"{blender_path}/blender --background --python {script_path} "
        f"{target} {res} {dest} > /dev/null 2>&1"
    )
    os.system(cmd)


def create_convex_hull(
    input_file: str,
    output_file: Optional[str] = None,
    target_faces: int = 500,
    manifold_path: Optional[str] = None,
    manifold_res: int = 5000,
    use_blender: bool = False,
    blender_path: Optional[str] = None,
    blender_res: int = 5,
) -> Path:
    """
    Create a convex hull initial mesh from a point cloud.

    Args:
        input_file: Path to input point cloud (.ply, .xyz, or .npts file).
        output_file: Path to output OBJ file. If None, uses input name with '_hull.obj'.
        target_faces: Target number of faces for the output mesh.
        manifold_path: Path to Manifold software build directory.
        manifold_res: Resolution for Manifold software.
        use_blender: Use Blender instead of Manifold for hull generation.
        blender_path: Path to Blender installation directory.
        blender_res: Resolution for Blender convex hull generation.

    Returns:
        Path to the output convex hull mesh.

    Raises:
        ImportError: If trimesh is not installed.
        FileNotFoundError: If input file or required software not found.
    """
    if not HAS_TRIMESH:
        raise ImportError(
            "trimesh is required for convex hull generation. "
            "Install it with: pip install trimesh"
        )

    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Determine output path
    if output_file is None:
        output_path = input_path.with_name(
            input_path.stem + "_hull.obj"
        )
    else:
        output_path = Path(output_file)

    # Set default paths
    if manifold_path is None:
        manifold_path = Path.home() / "code" / "Manifold" / "build"
    else:
        manifold_path = Path(manifold_path)

    if blender_path is not None:
        blender_path = Path(blender_path)

    # Validate software paths
    if use_blender:
        if blender_path is None or not (blender_path / "blender").exists():
            raise FileNotFoundError(
                "Blender not found. Provide --blender-path or use Manifold."
            )
    else:
        if not (manifold_path / "manifold").exists():
            raise FileNotFoundError(
                f"Manifold software not found at {manifold_path}. "
                "Install from https://github.com/hjwdzh/Manifold"
            )
        if not (manifold_path / "simplify").exists():
            raise FileNotFoundError(
                f"Simplify software not found at {manifold_path}"
            )

    # Read point cloud
    print(f"Reading point cloud from {input_path}...")
    xyz, _ = read_pts(str(input_path))

    # Generate convex hull
    print("Generating convex hull...")
    mesh = trimesh.convex.convex_hull(xyz[:, :3])
    vs, faces = mesh.vertices, mesh.faces

    # Export initial hull
    export(str(output_path), vs, faces)

    # Process with Manifold or Blender
    if use_blender:
        print(f"Processing with Blender (resolution: {blender_res})...")
        run_blender_hull(output_path, output_path, blender_res, blender_path)
    else:
        print(f"Processing with Manifold (resolution: {manifold_res})...")
        run_manifold(output_path, manifold_res, manifold_path)

    # Check face count and simplify if needed
    num_faces = count_faces(output_path)
    if use_blender:
        num_faces //= 2
    num_faces = int(num_faces)

    if num_faces < target_faces:
        software = "blender" if use_blender else "manifold"
        warnings.warn(
            f"Only {num_faces} faces were generated by {software}. "
            f"Try increasing --{software}-res to achieve the target of {target_faces} faces."
        )
    else:
        print(f"Simplifying to {target_faces} faces...")
        run_simplify(output_path, target_faces, manifold_path)

    print(f"Done! Convex hull saved to {output_path}")
    return output_path


def main():
    """Command-line interface for convex hull generation."""
    parser = argparse.ArgumentParser(
        description="Create a convex hull initial mesh from a point cloud"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input point cloud (.ply, .xyz, or .npts)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to output OBJ file (default: <input>_hull.obj)",
    )
    parser.add_argument(
        "--faces",
        "-f",
        type=int,
        required=True,
        help="Target number of faces for the convex hull",
    )
    parser.add_argument(
        "--manifold-path",
        type=str,
        default=None,
        help="Path to Manifold software build directory",
    )
    parser.add_argument(
        "--manifold-res",
        type=int,
        default=5000,
        help="Resolution for Manifold software (default: 5000)",
    )
    parser.add_argument(
        "--blender",
        action="store_true",
        help="Use Blender instead of Manifold for hull generation",
    )
    parser.add_argument(
        "--blender-path",
        type=str,
        default=None,
        help="Path to Blender installation directory",
    )
    parser.add_argument(
        "--blender-res",
        type=int,
        default=5,
        help="Resolution for Blender hull generation (default: 5)",
    )

    args = parser.parse_args()

    create_convex_hull(
        input_file=args.input,
        output_file=args.output,
        target_faces=args.faces,
        manifold_path=args.manifold_path,
        manifold_res=args.manifold_res,
        use_blender=args.blender,
        blender_path=args.blender_path,
        blender_res=args.blender_res,
    )


if __name__ == "__main__":
    main()
