from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from gpu4pyscf.dft import rks as gpu_rks
from pyscf import gto


WATER_MONOMER = (
    ("O", (0.0000, 0.0000, 0.0000)),
    ("H", (0.0000, 0.7586, 0.5858)),
    ("H", (0.0000, -0.7586, 0.5858)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "GPU4PySCF baseline that follows the paper's aligned end-to-end "
            "benchmark settings: B3LYP, tight SCF convergence, and exact "
            "analytical J/K integrals without density fitting."
        )
    )
    parser.add_argument(
        "--xyz",
        type=Path,
        help="Optional XYZ file. When provided, it overrides the built-in system.",
    )
    parser.add_argument(
        "--system",
        choices=("water", "water-cluster"),
        default="water-cluster",
        help="Built-in system to use when --xyz is not provided.",
    )
    parser.add_argument(
        "--waters",
        type=int,
        default=8,
        help="Number of water molecules for --system water-cluster.",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=3.0,
        help="Water cluster lattice spacing in Angstrom.",
    )
    parser.add_argument("--basis", default="def2-tzvp")
    parser.add_argument("--xc", default="B3LYP")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--spin", type=int, default=0)
    parser.add_argument(
        "--conv-tol",
        type=float,
        default=1e-7,
        help="SCF convergence tolerance from the paper's aligned settings.",
    )
    parser.add_argument("--max-cycle", type=int, default=50)
    parser.add_argument(
        "--grids-level",
        type=int,
        default=5,
        help=(
            "DFT grid level. The paper specifies exact J/K and B3LYP, but not a "
            "grid level; this keeps a high-accuracy default configurable."
        ),
    )
    parser.add_argument(
        "--avg-window",
        type=int,
        default=10,
        help="Average up to this many SCF iterations after excluding the first.",
    )
    parser.add_argument("--verbose", type=int, default=4)
    return parser.parse_args()


def load_xyz(path: Path) -> str:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"XYZ file is empty: {path}")

    try:
        natoms = int(lines[0])
        body = lines[2 : 2 + natoms]
    except ValueError:
        body = lines

    atoms: list[str] = []
    for line in body:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid XYZ line: {line}")
        symbol = parts[0]
        x, y, z = parts[1:4]
        atoms.append(f"{symbol} {x} {y} {z}")
    return "; ".join(atoms)


def build_water_cluster(num_waters: int, spacing: float) -> str:
    if num_waters < 1:
        raise ValueError("--waters must be at least 1")

    side = math.ceil(num_waters ** (1.0 / 3.0))
    shift = (side - 1) * spacing / 2.0
    atoms: list[str] = []

    count = 0
    for ix in range(side):
        for iy in range(side):
            for iz in range(side):
                if count >= num_waters:
                    return "; ".join(atoms)
                dx = ix * spacing - shift
                dy = iy * spacing - shift
                dz = iz * spacing - shift
                for symbol, (x, y, z) in WATER_MONOMER:
                    atoms.append(f"{symbol} {x + dx:.6f} {y + dy:.6f} {z + dz:.6f}")
                count += 1

    return "; ".join(atoms)


def build_atom_spec(args: argparse.Namespace) -> tuple[str, str]:
    if args.xyz is not None:
        return load_xyz(args.xyz), f"xyz:{args.xyz}"
    if args.system == "water":
        return build_water_cluster(1, args.spacing), "water"
    return build_water_cluster(args.waters, args.spacing), f"water-cluster:{args.waters}"


class IterationTimer:
    def __init__(self) -> None:
        self._last = time.perf_counter()
        self._seen_cycles: set[int] = set()
        self.iteration_seconds: list[float] = []

    def __call__(self, envs: dict) -> None:
        cycle = int(envs.get("cycle", -1)) + 1
        if cycle <= 0 or cycle in self._seen_cycles:
            return
        now = time.perf_counter()
        self.iteration_seconds.append(now - self._last)
        self._last = now
        self._seen_cycles.add(cycle)


def main() -> None:
    args = parse_args()
    atom_spec, system_label = build_atom_spec(args)

    mol = gto.M(
        atom=atom_spec,
        basis=args.basis,
        charge=args.charge,
        spin=args.spin,
        unit="Angstrom",
        verbose=args.verbose,
    )

    mf = gpu_rks.RKS(mol, xc=args.xc)
    mf.conv_tol = args.conv_tol
    mf.max_cycle = args.max_cycle
    mf.grids.level = args.grids_level
    mf.direct_scf = True

    # The paper's baseline uses analytical J/K integrals without density fitting.
    if hasattr(mf, "with_df"):
        mf.with_df = None

    timer = IterationTimer()
    start = time.perf_counter()
    energy = mf.kernel(callback=timer)
    total_seconds = time.perf_counter() - start

    completed_cycles = len(timer.iteration_seconds)
    post_warmup = timer.iteration_seconds[1 : 1 + args.avg_window]
    if not post_warmup:
        post_warmup = timer.iteration_seconds[1:] or timer.iteration_seconds

    summary = {
        "system": system_label,
        "natm": mol.natm,
        "nao": mol.nao_nr(),
        "basis": args.basis,
        "xc": args.xc,
        "charge": args.charge,
        "spin": args.spin,
        "conv_tol": args.conv_tol,
        "grid_level": args.grids_level,
        "direct_scf": mf.direct_scf,
        "density_fitting": False,
        "converged": bool(mf.converged),
        "energy_hartree": float(energy),
        "total_seconds": total_seconds,
        "completed_cycles": completed_cycles,
        "iteration_seconds": timer.iteration_seconds,
        "paper_style_avg_iteration_seconds": (
            sum(post_warmup) / len(post_warmup) if post_warmup else None
        ),
        "paper_style_avg_window_used": len(post_warmup),
    }

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
