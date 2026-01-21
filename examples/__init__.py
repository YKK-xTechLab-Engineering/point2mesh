"""
Point2Mesh example scripts.

This directory contains standalone example scripts for:
- Running Point2Mesh reconstruction on various models
- Downloading example data
- Preprocessing meshes (sampling, convex hull generation)

Usage:
    Run scripts directly from the examples directory:

        python examples/download_data.py
        python examples/run_guitar.py
        python examples/sample_mesh.py input.obj output.ply --num-samples 75000

Available reconstruction examples:
    - run_guitar.py: Standard guitar model
    - run_bull.py: Bull model with pooling
    - run_giraffe.py: Giraffe model
    - run_tiki.py: Tiki statue model
    - run_triceratops.py: Triceratops model
    - run_noisy_guitar.py: Guitar with noisy input (robustness demo)
    - run_letter_g.py: Letter G with beamgap loss (thin structures)
"""
