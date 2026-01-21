"""
Sample points from a mesh surface.

This script samples points and normals from a ground-truth mesh,
which can then be used as input to Point2Mesh reconstruction.
"""

import argparse
from pathlib import Path

import torch
from pytorch3d.io import load_objs_as_meshes, save_ply
from pytorch3d.ops import sample_points_from_meshes


def sample_mesh(
    input_obj: str,
    output_ply: str,
    num_samples: int = 75000,
    device: str = "cpu",
) -> None:
    """
    Sample points and normals from a mesh surface.

    Args:
        input_obj: Path to input OBJ mesh file.
        output_ply: Path to output PLY file with sampled points.
        num_samples: Number of points to sample. Defaults to 75000.
        device: Device to use for computation. Defaults to "cpu".
    """
    device = torch.device(device)

    print(f"Loading mesh from {input_obj}...")
    mesh = load_objs_as_meshes([input_obj], device=device)

    print(f"Sampling {num_samples} points from mesh surface...")
    xyz, normals = sample_points_from_meshes(
        mesh, num_samples=num_samples, return_normals=True
    )

    print(f"Saving point cloud to {output_ply}...")
    save_ply(output_ply, verts=xyz[0, :], verts_normals=normals[0, :])

    print("Done!")


def main():
    """Command-line interface for mesh sampling."""
    parser = argparse.ArgumentParser(
        description="Sample points and normals from a mesh surface"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input OBJ mesh file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path to output PLY file",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        type=int,
        default=75000,
        help="Number of points to sample (default: 75000)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use (default: cpu)",
    )

    args = parser.parse_args()
    sample_mesh(
        input_obj=args.input,
        output_ply=args.output,
        num_samples=args.num_samples,
        device=args.device,
    )


if __name__ == "__main__":
    main()
