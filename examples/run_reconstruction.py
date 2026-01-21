"""
Run Point2Mesh reconstruction.

This module provides a programmatic interface to run Point2Mesh reconstruction
with custom parameters, as well as pre-configured examples.
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ReconstructionConfig:
    """Configuration for Point2Mesh reconstruction."""

    # Required paths
    input_pc: str
    initial_mesh: str
    save_path: str

    # Training parameters
    iterations: int = 6000
    lr: float = 1.1e-4
    torch_seed: int = 5

    # Sampling parameters
    samples: int = 25000
    begin_samples: int = 15000
    upsamp: int = 1000
    max_faces: int = 10000

    # Network parameters
    convs: List[int] = field(default_factory=lambda: [16, 32, 64, 64, 128])
    pools: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    res_blocks: int = 3
    leaky_relu: float = 0.01
    init_weights: float = 0.002

    # Loss parameters
    ang_wt: float = 0.1
    local_non_uniform: float = 0.1

    # Beamgap parameters
    beamgap_iterations: int = 0
    beamgap_modulo: int = 1

    # Manifold parameters
    manifold_res: int = 100000
    manifold_always: bool = False

    # Mesh partitioning
    faces_to_part: List[int] = field(default_factory=lambda: [8000, 16000, 20000])
    overlap: int = 0
    global_step: bool = False
    transfer_data: bool = False

    # Other
    unoriented: bool = False
    export_interval: int = 100
    gpu: int = 0


def run_reconstruction(config: ReconstructionConfig) -> None:
    """
    Run Point2Mesh reconstruction with the given configuration.

    Args:
        config: ReconstructionConfig with all parameters.
    """
    # Build command-line arguments
    args = [
        sys.executable, "-m", "point2mesh.main",
        "--input-pc", config.input_pc,
        "--initial-mesh", config.initial_mesh,
        "--save-path", config.save_path,
        "--iterations", str(config.iterations),
        "--lr", str(config.lr),
        "--torch-seed", str(config.torch_seed),
        "--samples", str(config.samples),
        "--begin-samples", str(config.begin_samples),
        "--upsamp", str(config.upsamp),
        "--max-faces", str(config.max_faces),
        "--convs", *[str(c) for c in config.convs],
        "--pools", *[str(p) for p in config.pools],
        "--res-blocks", str(config.res_blocks),
        "--leaky-relu", str(config.leaky_relu),
        "--init-weights", str(config.init_weights),
        "--ang-wt", str(config.ang_wt),
        "--local-non-uniform", str(config.local_non_uniform),
        "--beamgap-iterations", str(config.beamgap_iterations),
        "--beamgap-modulo", str(config.beamgap_modulo),
        "--manifold-res", str(config.manifold_res),
        "--faces-to-part", *[str(f) for f in config.faces_to_part],
        "--overlap", str(config.overlap),
        "--export-interval", str(config.export_interval),
        "--gpu", str(config.gpu),
    ]

    if config.manifold_always:
        args.append("--manifold-always")
    if config.global_step:
        args.append("--global-step")
    if config.transfer_data:
        args.append("--transfer-data")
    if config.unoriented:
        args.append("--unoriented")

    # Import and run directly instead of subprocess for better integration
    import torch
    from point2mesh.models.layers.mesh import Mesh, PartMesh
    from point2mesh.models.networks import init_net, sample_surface, local_nonuniform_penalty
    from point2mesh import utils
    from point2mesh.models.losses import chamfer_distance, BeamGapLoss
    import numpy as np
    import time
    import os

    # Set up
    torch.manual_seed(config.torch_seed)
    device = torch.device(f'cuda:{config.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Create save directory
    os.makedirs(config.save_path, exist_ok=True)

    # Save config
    with open(os.path.join(config.save_path, 'config.txt'), 'w') as f:
        for k, v in vars(config).items():
            f.write(f'{k}: {v}\n')

    # Load initial mesh
    mesh = Mesh(config.initial_mesh, device=device, hold_history=True)

    # Load input point cloud
    input_xyz, input_normals = utils.read_pts(config.input_pc)
    input_xyz /= mesh.scale
    input_xyz += mesh.translations[None, :]
    input_xyz = torch.Tensor(input_xyz).float().to(device)[None, :, :]
    input_normals = torch.Tensor(input_normals).float().to(device)[None, :, :]

    # Helper to get number of parts
    def get_num_parts(num_faces):
        lookup = [1, 2, 4, 8]
        return lookup[np.digitize(num_faces, config.faces_to_part, right=True)]

    # Helper to get number of samples
    def get_num_samples(cur_iter):
        slope = (config.samples - config.begin_samples) / int(0.8 * config.upsamp)
        return int(slope * min(cur_iter, 0.8 * config.upsamp)) + config.begin_samples

    # Initialize network
    part_mesh = PartMesh(mesh, num_parts=get_num_parts(len(mesh.faces)), bfs_depth=config.overlap)
    print(f'Number of parts: {part_mesh.n_submeshes}')
    net, optimizer, rand_verts, scheduler = init_net(mesh, part_mesh, device, config)

    beamgap_loss = BeamGapLoss(device)
    if config.beamgap_iterations > 0:
        print('Beamgap loss enabled')
        beamgap_loss.update_pm(part_mesh, torch.cat([input_xyz, input_normals], dim=-1))

    # Training loop
    for i in range(config.iterations):
        num_samples = get_num_samples(i % config.upsamp)
        if config.global_step:
            optimizer.zero_grad()
        start_time = time.time()

        for part_i, est_verts in enumerate(net(rand_verts, part_mesh)):
            if not config.global_step:
                optimizer.zero_grad()
            part_mesh.update_verts(est_verts[0], part_i)
            num_samples = get_num_samples(i % config.upsamp)
            recon_xyz, recon_normals = sample_surface(
                part_mesh.main_mesh.faces,
                part_mesh.main_mesh.vs.unsqueeze(0),
                num_samples
            )
            recon_xyz, recon_normals = recon_xyz.float(), recon_normals.float()

            xyz_chamfer_loss, normals_chamfer_loss = chamfer_distance(
                recon_xyz, input_xyz,
                x_normals=recon_normals, y_normals=input_normals,
                unoriented=config.unoriented
            )

            if (i < config.beamgap_iterations) and (i % config.beamgap_modulo == 0):
                loss = beamgap_loss(part_mesh, part_i)
            else:
                loss = xyz_chamfer_loss + (config.ang_wt * normals_chamfer_loss)

            if config.local_non_uniform > 0:
                loss += config.local_non_uniform * local_nonuniform_penalty(part_mesh.main_mesh).float()

            loss.backward()
            if not config.global_step:
                optimizer.step()
                scheduler.step()
            part_mesh.main_mesh.vs.detach_()

        if config.global_step:
            optimizer.step()
            scheduler.step()
        end_time = time.time()

        if i % 1 == 0:
            print(f'{os.path.basename(config.input_pc)}; iter: {i}/{config.iterations}; '
                  f'loss: {loss.item():.4f}; samples: {num_samples}; time: {end_time - start_time:.2f}s')

        if i % config.export_interval == 0 and i > 0:
            print(f'Exporting reconstruction... LR: {optimizer.param_groups[0]["lr"]:.6f}')
            with torch.no_grad():
                part_mesh.export(os.path.join(config.save_path, f'recon_iter_{i}.obj'))

        if (i > 0 and (i + 1) % config.upsamp == 0):
            mesh = part_mesh.main_mesh
            num_faces = int(np.clip(len(mesh.faces) * 1.5, len(mesh.faces), config.max_faces))

            if num_faces > len(mesh.faces) or config.manifold_always:
                mesh = utils.manifold_upsample(
                    mesh, config.save_path, Mesh,
                    num_faces=min(num_faces, config.max_faces),
                    res=config.manifold_res, simplify=True
                )
                part_mesh = PartMesh(mesh, num_parts=get_num_parts(len(mesh.faces)), bfs_depth=config.overlap)
                print(f'Upsampled to {len(mesh.faces)} faces; parts: {part_mesh.n_submeshes}')
                net, optimizer, rand_verts, scheduler = init_net(mesh, part_mesh, device, config)
                if i < config.beamgap_iterations:
                    print('Beamgap updated')
                    beamgap_loss.update_pm(part_mesh, input_xyz)

    # Final export
    with torch.no_grad():
        mesh.export(os.path.join(config.save_path, 'final_recon.obj'))
    print(f'Done! Final reconstruction saved to {config.save_path}/final_recon.obj')


# Pre-configured examples
EXAMPLES = {
    "guitar": ReconstructionConfig(
        input_pc="./data/guitar.ply",
        initial_mesh="./data/guitar_initmesh.obj",
        save_path="./checkpoints/guitar",
        iterations=6000,
    ),
    "bull": ReconstructionConfig(
        input_pc="./data/bull.ply",
        initial_mesh="./data/bull_initmesh.obj",
        save_path="./checkpoints/bull",
        iterations=6000,
        pools=[0.1, 0.0, 0.0, 0.0],
    ),
    "giraffe": ReconstructionConfig(
        input_pc="./data/giraffe.ply",
        initial_mesh="./data/giraffe_initmesh.obj",
        save_path="./checkpoints/giraffe",
        iterations=6000,
    ),
    "tiki": ReconstructionConfig(
        input_pc="./data/tiki.ply",
        initial_mesh="./data/tiki_initmesh.obj",
        save_path="./checkpoints/tiki",
        iterations=6000,
    ),
    "triceratops": ReconstructionConfig(
        input_pc="./data/triceratops.ply",
        initial_mesh="./data/triceratops_initmesh.obj",
        save_path="./checkpoints/triceratops",
        iterations=6000,
    ),
    "noisy_guitar": ReconstructionConfig(
        input_pc="./data/noisy_guitar.ply",
        initial_mesh="./data/guitar_initmesh.obj",
        save_path="./checkpoints/noisy_guitar",
        iterations=6000,
    ),
    "letter_g": ReconstructionConfig(
        input_pc="./data/g.ply",
        initial_mesh="./data/g_initmesh.obj",
        save_path="./checkpoints/g",
        iterations=3000,
        lr=0.0001,
        upsamp=100,
        beamgap_iterations=800,
        beamgap_modulo=2,
        manifold_res=4000,
        convs=[64, 64, 64, 128],
        pools=[0.0, 0.0, 0.0],
        max_faces=10000,
        manifold_always=True,
    ),
}


def main():
    """Command-line interface for running reconstruction examples."""
    parser = argparse.ArgumentParser(
        description="Run Point2Mesh reconstruction"
    )
    parser.add_argument(
        "--example",
        type=str,
        choices=list(EXAMPLES.keys()),
        help="Run a pre-configured example",
    )
    parser.add_argument(
        "--input-pc",
        type=str,
        help="Path to input point cloud",
    )
    parser.add_argument(
        "--initial-mesh",
        type=str,
        help="Path to initial mesh",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        help="Path to save results",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=6000,
        help="Number of iterations (default: 6000)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device to use (default: 0)",
    )
    parser.add_argument(
        "--list-examples",
        action="store_true",
        help="List available examples and exit",
    )

    args = parser.parse_args()

    if args.list_examples:
        print("Available examples:")
        for name, config in EXAMPLES.items():
            print(f"  {name}: {config.input_pc}")
        return

    if args.example:
        config = EXAMPLES[args.example]
        if args.gpu != 0:
            config.gpu = args.gpu
        if args.iterations != 6000:
            config.iterations = args.iterations
    elif args.input_pc and args.initial_mesh and args.save_path:
        config = ReconstructionConfig(
            input_pc=args.input_pc,
            initial_mesh=args.initial_mesh,
            save_path=args.save_path,
            iterations=args.iterations,
            gpu=args.gpu,
        )
    else:
        parser.error("Either --example or (--input-pc, --initial-mesh, --save-path) required")

    run_reconstruction(config)


if __name__ == "__main__":
    main()
