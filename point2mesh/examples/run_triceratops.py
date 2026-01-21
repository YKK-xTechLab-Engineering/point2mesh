#!/usr/bin/env python
"""
Run Point2Mesh reconstruction on the triceratops example.

Usage:
    python -m point2mesh.examples.run_triceratops
"""

from .run_reconstruction import EXAMPLES, run_reconstruction


def main():
    """Run triceratops reconstruction with default parameters."""
    print("=" * 60)
    print("Point2Mesh: Triceratops Reconstruction")
    print("=" * 60)
    run_reconstruction(EXAMPLES["triceratops"])


if __name__ == "__main__":
    main()
