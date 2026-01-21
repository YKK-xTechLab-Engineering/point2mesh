#!/usr/bin/env python
"""
Run Point2Mesh reconstruction on the giraffe example.

Usage:
    python examples/run_giraffe.py
"""

from run_reconstruction import EXAMPLES, run_reconstruction


def main():
    """Run giraffe reconstruction with default parameters."""
    print("=" * 60)
    print("Point2Mesh: Giraffe Reconstruction")
    print("=" * 60)
    run_reconstruction(EXAMPLES["giraffe"])


if __name__ == "__main__":
    main()
