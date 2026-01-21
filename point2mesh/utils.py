import torch
import numpy as np
import os
import uuid
import glob
import shutil
from pathlib import Path


def get_manifold_bin_dir():
    """
    Get the directory containing the manifold executables.

    Checks in this order:
    1. MANIFOLD_DIR environment variable (for custom installations)
    2. CONDA_PREFIX/bin (when installed via pixi/conda)
    3. Package bin directory (for pip-installed wheels)
    """
    # Check for environment variable override
    env_dir = os.environ.get('MANIFOLD_DIR')
    if env_dir and os.path.isdir(env_dir):
        return env_dir

    # Check CONDA_PREFIX (pixi/conda environment)
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        bin_dir = os.path.join(conda_prefix, 'bin')
        if os.path.isfile(os.path.join(bin_dir, 'manifold')):
            return bin_dir

    # Check bundled executables (for pip wheels)
    package_dir = Path(__file__).parent
    bin_dir = package_dir / 'bin'
    if bin_dir.is_dir() and (bin_dir / 'manifold').is_file():
        return str(bin_dir)

    raise FileNotFoundError(
        "Manifold executables not found. Either:\n"
        "1. Install via pixi (manifold package will be built automatically)\n"
        "2. Set the MANIFOLD_DIR environment variable to your Manifold build directory\n"
        "3. Ensure CONDA_PREFIX is set and contains bin/manifold"
    )


def get_manifold_executable():
    """Get the path to the manifold executable."""
    bin_dir = get_manifold_bin_dir()
    manifold_path = os.path.join(bin_dir, 'manifold')
    if not os.path.exists(manifold_path):
        raise FileNotFoundError(f'manifold executable not found at {manifold_path}')
    return manifold_path


def get_simplify_executable():
    """Get the path to the simplify executable."""
    bin_dir = get_manifold_bin_dir()
    simplify_path = os.path.join(bin_dir, 'simplify')
    if not os.path.exists(simplify_path):
        raise FileNotFoundError(f'simplify executable not found at {simplify_path}')
    return simplify_path


def manifold_upsample(mesh, save_path, Mesh, num_faces=2000, res=3000, simplify=True):
    """
    Upsample a mesh using the Manifold algorithm.

    Args:
        mesh: Input mesh object with export method
        save_path: Directory to save intermediate files
        Mesh: Mesh class to create output mesh
        num_faces: Target number of faces after simplification
        res: Resolution for manifold algorithm (default: 3000)
        simplify: Whether to simplify the mesh after manifold processing

    Returns:
        Upsampled mesh object
    """
    # export before upsample
    fname = os.path.join(save_path, 'recon_{}.obj'.format(len(mesh.faces)))
    mesh.export(fname)

    temp_file = os.path.join(save_path, random_file_name('obj'))
    opts = ' ' + str(res) if res is not None else ''

    manifold_path = get_manifold_executable()
    cmd = "{} {} {}".format(manifold_path, fname, temp_file + opts)
    os.system(cmd)

    if simplify:
        simplify_path = get_simplify_executable()
        cmd = "{} -i {} -o {} -f {}".format(simplify_path, temp_file, temp_file, num_faces)
        os.system(cmd)

    m_out = Mesh(temp_file, hold_history=True, device=mesh.device)
    fname = os.path.join(save_path, 'recon_{}_after.obj'.format(len(m_out.faces)))
    m_out.export(fname)
    [os.remove(_) for _ in list(glob.glob(os.path.splitext(temp_file)[0] + '*'))]
    return m_out


def read_pts(pts_file):
    '''
    :param pts_file: file path of a plain text list of points
    such that a particular line has 6 float values: x, y, z, nx, ny, nz
    which is typical for (plaintext) .ply or .xyz
    :return: xyz, normals
    '''
    xyz, normals = [], []
    with open(pts_file, 'r') as f:
        # line = f.readline()
        spt = f.read().split('\n')
        # while line:
        for line in spt:
            parts = line.strip().split(' ')
            try:
                x = np.array(parts, dtype=np.float32)
                xyz.append(x[:3])
                normals.append(x[3:])
            except:
                pass
    return np.array(xyz, dtype=np.float32), np.array(normals, dtype=np.float32)


def load_obj(file):
    vs, faces = [], []
    f = open(file)
    for line in f:
        line = line.strip()
        splitted_line = line.split()
        if not splitted_line:
            continue
        elif splitted_line[0] == 'v':
            vs.append([float(v) for v in splitted_line[1:4]])
        elif splitted_line[0] == 'f':
            face_vertex_ids = [int(c.split('/')[0]) for c in splitted_line[1:]]
            assert len(face_vertex_ids) == 3
            face_vertex_ids = [(ind - 1) if (ind >= 0) else (len(vs) + ind)
                               for ind in face_vertex_ids]
            faces.append(face_vertex_ids)
    f.close()
    vs = np.asarray(vs)
    faces = np.asarray(faces, dtype=int)
    assert np.logical_and(faces >= 0, faces < len(vs)).all()
    return vs, faces


def export(file, vs, faces, vn=None, color=None):
    with open(file, 'w+') as f:
        for vi, v in enumerate(vs):
            if color is None:
                f.write("v %f %f %f\n" % (v[0], v[1], v[2]))
            else:
                f.write("v %f %f %f %f %f %f\n" % (v[0], v[1], v[2], color[vi][0], color[vi][1], color[vi][2]))
            if vn is not None:
                f.write("vn %f %f %f\n" % (vn[vi, 0], vn[vi, 1], vn[vi, 2]))
        for face in faces:
            f.write("f %d %d %d\n" % (face[0] + 1, face[1] + 1, face[2] + 1))


def random_file_name(ext, prefix='temp'):
    return f'{prefix}{uuid.uuid4()}.{ext}'
