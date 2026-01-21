"""
Download example data for Point2Mesh.

This script downloads the example point clouds and initial meshes
from the official Point2Mesh repository.
"""

import argparse
import os
import tarfile
import urllib.request
from pathlib import Path


DATA_URL = "https://www.dropbox.com/s/nn9hsww6mr0zccm/data.tar?dl=1"


def download_data(output_dir: str = "./data", verbose: bool = True) -> Path:
    """
    Download and extract Point2Mesh example data.

    Args:
        output_dir: Directory to extract data to. Defaults to "./data".
        verbose: Whether to print progress messages.

    Returns:
        Path to the extracted data directory.
    """
    output_path = Path(output_dir)
    tar_path = output_path.parent / "data.tar"

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Downloading Point2Mesh example data from {DATA_URL}...")

    # Download the tar file
    urllib.request.urlretrieve(DATA_URL, tar_path)

    if verbose:
        print(f"Extracting to {output_path.parent}...")

    # Extract the tar file
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=output_path.parent)

    # Remove the tar file
    os.remove(tar_path)

    if verbose:
        print(f"Done! Data extracted to {output_path}")
        print("\nAvailable examples:")
        if output_path.exists():
            for f in sorted(output_path.glob("*.ply")):
                print(f"  - {f.stem}")

    return output_path


def main():
    """Command-line interface for downloading data."""
    parser = argparse.ArgumentParser(
        description="Download Point2Mesh example data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Directory to extract data to (default: ./data)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )

    args = parser.parse_args()
    download_data(output_dir=args.output_dir, verbose=not args.quiet)


if __name__ == "__main__":
    main()
