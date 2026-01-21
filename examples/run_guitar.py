#!/usr/bin/env python
"""
Run Point2Mesh reconstruction on the guitar example.

Usage:
    python examples/run_guitar.py
"""

from run_reconstruction import EXAMPLES, run_reconstruction


def main():
    """Run guitar reconstruction with default parameters."""
    print("=" * 60)
    print("Point2Mesh: Guitar Reconstruction")
    print("=" * 60)
    run_reconstruction(EXAMPLES["guitar"])


if __name__ == "__main__":
    main()
