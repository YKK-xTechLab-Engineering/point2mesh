"""
Tests for Manifold integration in point2mesh.

These tests verify that:
1. The bundled Manifold executables are correctly located
2. The manifold executable can process meshes
3. The simplify executable can reduce mesh complexity
"""
import os
import subprocess
import pytest
from pathlib import Path

from point2mesh.utils import (
    get_manifold_bin_dir,
    get_manifold_executable,
    get_simplify_executable,
    load_obj,
    export,
)


class TestManifoldExecutableDiscovery:
    """Tests for finding the Manifold executables."""

    def test_get_manifold_bin_dir_exists(self):
        """Test that get_manifold_bin_dir returns a valid directory."""
        bin_dir = get_manifold_bin_dir()
        assert os.path.isdir(bin_dir), f"Manifold bin directory not found: {bin_dir}"

    def test_get_manifold_executable_exists(self):
        """Test that the manifold executable exists."""
        manifold_path = get_manifold_executable()
        assert os.path.isfile(manifold_path), f"manifold executable not found: {manifold_path}"
        assert os.access(manifold_path, os.X_OK), f"manifold is not executable: {manifold_path}"

    def test_get_simplify_executable_exists(self):
        """Test that the simplify executable exists."""
        simplify_path = get_simplify_executable()
        assert os.path.isfile(simplify_path), f"simplify executable not found: {simplify_path}"
        assert os.access(simplify_path, os.X_OK), f"simplify is not executable: {simplify_path}"

    def test_manifold_dir_env_override(self, temp_dir, monkeypatch):
        """Test that MANIFOLD_DIR environment variable overrides bundled path."""
        # Create dummy executables in temp dir
        manifold_path = os.path.join(temp_dir, "manifold")
        simplify_path = os.path.join(temp_dir, "simplify")
        Path(manifold_path).touch()
        Path(simplify_path).touch()
        os.chmod(manifold_path, 0o755)
        os.chmod(simplify_path, 0o755)

        monkeypatch.setenv("MANIFOLD_DIR", temp_dir)
        assert get_manifold_bin_dir() == temp_dir


class TestManifoldExecution:
    """Tests for running the Manifold executables."""

    def test_manifold_help_runs(self):
        """Test that manifold executable can run (may not have --help flag)."""
        manifold_path = get_manifold_executable()
        # Just verify it's executable - manifold doesn't have --help
        result = subprocess.run(
            [manifold_path],
            capture_output=True,
            timeout=10,
        )
        # It will return non-zero without args, but that's expected
        assert result.returncode != 127, "manifold executable not found by shell"

    def test_simplify_help_runs(self):
        """Test that simplify executable can run (may not have --help flag)."""
        simplify_path = get_simplify_executable()
        result = subprocess.run(
            [simplify_path],
            capture_output=True,
            timeout=10,
        )
        # It will return non-zero without args, but that's expected
        assert result.returncode != 127, "simplify executable not found by shell"

    def test_manifold_processes_cube(self, simple_cube_obj, temp_dir):
        """Test that manifold can process a simple cube mesh."""
        manifold_path = get_manifold_executable()
        output_path = os.path.join(temp_dir, "output.obj")

        result = subprocess.run(
            [manifold_path, simple_cube_obj, output_path, "1000"],
            capture_output=True,
            timeout=60,
        )

        assert result.returncode == 0, f"manifold failed: {result.stderr.decode()}"
        assert os.path.exists(output_path), "manifold did not create output file"

        # Verify output is valid OBJ
        vs, faces = load_obj(output_path)
        assert len(vs) > 0, "Output mesh has no vertices"
        assert len(faces) > 0, "Output mesh has no faces"

    def test_simplify_reduces_faces(self, simple_cube_obj, temp_dir):
        """Test that simplify can reduce mesh face count."""
        manifold_path = get_manifold_executable()
        simplify_path = get_simplify_executable()

        # First run manifold to get a higher-resolution mesh
        manifold_output = os.path.join(temp_dir, "manifold_output.obj")
        result = subprocess.run(
            [manifold_path, simple_cube_obj, manifold_output, "5000"],
            capture_output=True,
            timeout=60,
        )
        assert result.returncode == 0, f"manifold failed: {result.stderr.decode()}"

        # Load manifold output to check face count
        vs_before, faces_before = load_obj(manifold_output)

        # Now simplify the mesh
        simplified_output = os.path.join(temp_dir, "simplified.obj")
        target_faces = 100
        result = subprocess.run(
            [simplify_path, "-i", manifold_output, "-o", simplified_output, "-f", str(target_faces)],
            capture_output=True,
            timeout=60,
        )
        assert result.returncode == 0, f"simplify failed: {result.stderr.decode()}"
        assert os.path.exists(simplified_output), "simplify did not create output file"

        # Verify output has reduced faces
        vs_after, faces_after = load_obj(simplified_output)
        assert len(faces_after) <= len(faces_before), "simplify did not reduce face count"
        # Allow some tolerance since simplify may not hit exact target
        assert len(faces_after) <= target_faces * 1.5, f"simplify face count {len(faces_after)} far exceeds target {target_faces}"


class TestUtilityFunctions:
    """Tests for utility functions in utils.py."""

    def test_load_obj_cube(self, simple_cube_obj):
        """Test loading an OBJ file."""
        vs, faces = load_obj(simple_cube_obj)
        assert len(vs) == 8, f"Expected 8 vertices, got {len(vs)}"
        assert len(faces) == 12, f"Expected 12 faces, got {len(faces)}"

    def test_load_obj_tetrahedron(self, tetrahedron_obj):
        """Test loading a tetrahedron OBJ file."""
        vs, faces = load_obj(tetrahedron_obj)
        assert len(vs) == 4, f"Expected 4 vertices, got {len(vs)}"
        assert len(faces) == 4, f"Expected 4 faces, got {len(faces)}"

    def test_export_and_reload(self, temp_dir):
        """Test that exporting and reloading preserves mesh data."""
        import numpy as np

        # Create simple mesh data
        vs = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=int)

        # Export
        output_path = os.path.join(temp_dir, "triangle.obj")
        export(output_path, vs, faces)

        # Reload
        vs_loaded, faces_loaded = load_obj(output_path)

        np.testing.assert_array_almost_equal(vs, vs_loaded, decimal=5)
        np.testing.assert_array_equal(faces, faces_loaded)


class TestManifoldPipeline:
    """Integration tests for the full manifold pipeline."""

    def test_full_pipeline_cube(self, simple_cube_obj, temp_dir):
        """Test full manifold + simplify pipeline on a cube."""
        manifold_path = get_manifold_executable()
        simplify_path = get_simplify_executable()

        # Run manifold
        manifold_output = os.path.join(temp_dir, "step1.obj")
        subprocess.run(
            [manifold_path, simple_cube_obj, manifold_output, "2000"],
            check=True,
            capture_output=True,
            timeout=60,
        )

        # Run simplify
        final_output = os.path.join(temp_dir, "final.obj")
        subprocess.run(
            [simplify_path, "-i", manifold_output, "-o", final_output, "-f", "50"],
            check=True,
            capture_output=True,
            timeout=60,
        )

        # Verify output
        vs, faces = load_obj(final_output)
        assert len(vs) > 0
        assert len(faces) > 0
        assert len(faces) <= 75  # Allow some tolerance
