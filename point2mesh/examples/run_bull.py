#!/usr/bin/env python
"""
Run Point2Mesh reconstruction on the bull example.

Usage:
    python -m point2mesh.examples.run_bull
"""

from .run_reconstruction import EXAMPLES, run_reconstruction


def main():
    """Run bull reconstruction with default parameters."""
    print("=" * 60)
    print("Point2Mesh: Bull Reconstruction")
    print("=" * 60)
    run_reconstruction(EXAMPLES["bull"])


if __name__ == "__main__":
    main()
