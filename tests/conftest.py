import pytest
import tempfile
import os
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def simple_cube_obj(temp_dir):
    """Create a simple cube OBJ file for testing."""
    cube_path = os.path.join(temp_dir, "cube.obj")
    with open(cube_path, 'w') as f:
        f.write("""# Simple cube
v -1.0 -1.0 -1.0
v -1.0 -1.0  1.0
v -1.0  1.0 -1.0
v -1.0  1.0  1.0
v  1.0 -1.0 -1.0
v  1.0 -1.0  1.0
v  1.0  1.0 -1.0
v  1.0  1.0  1.0
f 1 3 4
f 1 4 2
f 5 6 8
f 5 8 7
f 1 2 6
f 1 6 5
f 3 7 8
f 3 8 4
f 1 5 7
f 1 7 3
f 2 4 8
f 2 8 6
""")
    return cube_path


@pytest.fixture
def tetrahedron_obj(temp_dir):
    """Create a tetrahedron OBJ file for testing."""
    tetra_path = os.path.join(temp_dir, "tetrahedron.obj")
    with open(tetra_path, 'w') as f:
        f.write("""# Simple tetrahedron
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.5 0.866 0.0
v 0.5 0.289 0.816
f 1 2 3
f 1 4 2
f 2 4 3
f 3 4 1
""")
    return tetra_path
