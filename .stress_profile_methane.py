import json
import time
from pathlib import Path

from b3lyp import (
    build_atoms,
    build_basis_functions,
    build_grid,
    build_integrals,
    nuclear_repulsion_energy,
    parse_orca_basis,
    restricted_scf,
)

geometry = [
    ("C", (0.000000, 0.000000, 0.000000)),
    ("H", (0.629118, 0.629118, 0.629118)),
    ("H", (-0.629118, -0.629118, 0.629118)),
    ("H", (-0.629118, 0.629118, -0.629118)),
    ("H", (0.629118, -0.629118, -0.629118)),
]

start = time.perf_counter()
basis = parse_orca_basis(Path("def2-TZVP.orca"))
atoms, nelec = build_atoms(geometry, 0)
basis_functions = build_basis_functions(atoms, basis)
after_basis = time.perf_counter()
overlap, core_h, eri = build_integrals(atoms, basis_functions)
after_integrals = time.perf_counter()
grid_points, grid_weights = build_grid(atoms, num_radial=16, num_angular=26, partition_power=4.0)
after_grid = time.perf_counter()
scf = restricted_scf(
    method="b3lyp",
    core_h=core_h,
    overlap=overlap,
    eri=eri,
    nelec=nelec,
    e_nuc=nuclear_repulsion_energy(atoms),
    basis_functions=basis_functions,
    grid_points=grid_points,
    grid_weights=grid_weights,
    max_cycle=8,
    conv_tol=1e-8,
    density_tol=1e-6,
    verbose=False,
)
after_scf = time.perf_counter()
print(json.dumps({
    "system": "methane",
    "atoms": len(atoms),
    "basis_functions": len(basis_functions),
    "grid_points": int(len(grid_weights)),
    "basis_seconds": after_basis - start,
    "integrals_seconds": after_integrals - after_basis,
    "grid_seconds": after_grid - after_integrals,
    "scf_seconds": after_scf - after_grid,
    "total_seconds": after_scf - start,
    "converged": scf["converged"],
    "cycles": scf["cycles"],
    "energy_hartree": scf["energy_hartree"],
}, indent=2))
