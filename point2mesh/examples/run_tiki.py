#!/usr/bin/env python
"""
Run Point2Mesh reconstruction on the tiki example.

Usage:
    python -m point2mesh.examples.run_tiki
"""

from .run_reconstruction import EXAMPLES, run_reconstruction


def main():
    """Run tiki reconstruction with default parameters."""
    print("=" * 60)
    print("Point2Mesh: Tiki Reconstruction")
    print("=" * 60)
    run_reconstruction(EXAMPLES["tiki"])


if __name__ == "__main__":
    main()
