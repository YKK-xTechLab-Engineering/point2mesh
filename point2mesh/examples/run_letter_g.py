#!/usr/bin/env python
"""
Run Point2Mesh reconstruction on the letter G example.

This example uses beamgap loss and different network parameters
for reconstructing thin structures.

Usage:
    python -m point2mesh.examples.run_letter_g
"""

from .run_reconstruction import EXAMPLES, run_reconstruction


def main():
    """Run letter G reconstruction with specialized parameters."""
    print("=" * 60)
    print("Point2Mesh: Letter G Reconstruction")
    print("=" * 60)
    print("Note: This example uses beamgap loss for thin structures")
    run_reconstruction(EXAMPLES["letter_g"])


if __name__ == "__main__":
    main()
