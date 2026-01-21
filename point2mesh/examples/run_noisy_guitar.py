#!/usr/bin/env python
"""
Run Point2Mesh reconstruction on the noisy guitar example.

This example demonstrates Point2Mesh's robustness to noise in the input.

Usage:
    python -m point2mesh.examples.run_noisy_guitar
"""

from .run_reconstruction import EXAMPLES, run_reconstruction


def main():
    """Run noisy guitar reconstruction with default parameters."""
    print("=" * 60)
    print("Point2Mesh: Noisy Guitar Reconstruction")
    print("=" * 60)
    run_reconstruction(EXAMPLES["noisy_guitar"])


if __name__ == "__main__":
    main()
