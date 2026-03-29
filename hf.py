from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cupy as cp
import numpy as np


ANGSTROM_TO_BOHR = 1.8897259886
ATOM_NAMES = {
    "H": "H",
    "HYDROGEN": "H",
    "O": "O",
    "OXYGEN": "O",
}
ATOMIC_NUMBERS = {
    "H": 1,
    "O": 8,
}
WATER = (
    ("O", (0.0000, 0.0000, 0.0000)),
    ("H", (0.0000, 0.7586, 0.5858)),
    ("H", (0.0000, -0.7586, 0.5858)),
)
H2 = (
    ("H", (0.0000, 0.0000, -0.3700)),
    ("H", (0.0000, 0.0000, 0.3700)),
)


@dataclass(frozen=True)
class ShellSpec:
    shell_type: str
    exponents: tuple[float, ...]
    coeffs: tuple[float, ...]


@dataclass(frozen=True)
class Atom:
    symbol: str
    charge: int
    coord: np.ndarray


@dataclass(frozen=True)
class BasisFunction:
    center: np.ndarray
    ang: tuple[int, int, int]
    exponents: np.ndarray
    coeffs: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Minimal restricted Hartree-Fock SCF from scratch for H/O systems "
            "using analytic STO-3G integrals and a CuPy-backed SCF loop."
        )
    )
    parser.add_argument(
        "--basis",
        type=Path,
        default=Path(__file__).with_name("STO-3G.orca"),
        help="Path to an ORCA-format STO-3G basis file.",
    )
    parser.add_argument(
        "--xyz",
        type=Path,
        help="Optional XYZ file. Supports atoms that exist in the basis file.",
    )
    parser.add_argument(
        "--system",
        choices=("water", "h2"),
        default="water",
        help="Built-in geometry to run when --xyz is not provided.",
    )
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--max-cycle", type=int, default=50)
    parser.add_argument("--conv-tol", type=float, default=1e-9)
    parser.add_argument("--density-tol", type=float, default=1e-7)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_xyz(path: Path) -> list[tuple[str, tuple[float, float, float]]]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"XYZ file is empty: {path}")

    try:
        natoms = int(lines[0])
        body = lines[2 : 2 + natoms]
    except ValueError:
        body = lines

    atoms: list[tuple[str, tuple[float, float, float]]] = []
    for line in body:
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"Invalid XYZ line: {line}")
        symbol = normalize_symbol(fields[0])
        coord = (float(fields[1]), float(fields[2]), float(fields[3]))
        atoms.append((symbol, coord))
    return atoms


def get_geometry(args: argparse.Namespace) -> list[tuple[str, tuple[float, float, float]]]:
    if args.xyz is not None:
        return load_xyz(args.xyz)
    if args.system == "h2":
        return list(H2)
    return list(WATER)


def normalize_symbol(label: str) -> str:
    key = label.strip().upper()
    if key not in ATOM_NAMES:
        raise ValueError(f"Unsupported atom label: {label}")
    return ATOM_NAMES[key]


def parse_orca_basis(path: Path) -> dict[str, list[ShellSpec]]:
    lines = path.read_text().splitlines()
    in_data = False
    current_atom: str | None = None
    basis: dict[str, list[ShellSpec]] = {}
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        i += 1

        if not line or line.startswith("!"):
            continue
        if line == "$DATA":
            in_data = True
            continue
        if line == "$END":
            break
        if not in_data:
            continue

        fields = line.split()
        head = fields[0].upper()

        if head in ATOM_NAMES:
            current_atom = normalize_symbol(head)
            basis.setdefault(current_atom, [])
            continue

        if current_atom is None:
            raise ValueError(f"Found shell before atom header in {path}")
        if head not in {"S", "L"}:
            raise ValueError(f"Unsupported shell type in {path}: {head}")

        nprims = int(fields[1])
        exponents: list[float] = []
        coeffs: list[float] = []
        coeffs_p: list[float] | None = [] if head == "L" else None
        for _ in range(nprims):
            row = lines[i].split()
            i += 1
            exponents.append(float(row[1]))
            coeffs.append(float(row[2]))
            if coeffs_p is not None:
                coeffs_p.append(float(row[3]))

        basis[current_atom].append(
            ShellSpec(
                shell_type="S",
                exponents=tuple(exponents),
                coeffs=tuple(coeffs),
            )
        )
        if coeffs_p is not None:
            basis[current_atom].append(
                ShellSpec(
                    shell_type="P",
                    exponents=tuple(exponents),
                    coeffs=tuple(coeffs_p),
                )
            )

    return basis


def build_atoms(
    geometry: list[tuple[str, tuple[float, float, float]]],
    charge: int,
) -> tuple[list[Atom], int]:
    atoms: list[Atom] = []
    total_nuclear_charge = 0
    for symbol, coord_angstrom in geometry:
        coord_bohr = np.asarray(coord_angstrom, dtype=float) * ANGSTROM_TO_BOHR
        z = ATOMIC_NUMBERS[symbol]
        atoms.append(Atom(symbol=symbol, charge=z, coord=coord_bohr))
        total_nuclear_charge += z

    nelec = total_nuclear_charge - charge
    if nelec <= 0:
        raise ValueError("Total electron count must be positive")
    if nelec % 2 != 0:
        raise ValueError("This minimal implementation only supports closed-shell systems")
    return atoms, nelec


def build_basis_functions(
    atoms: list[Atom],
    parsed_basis: dict[str, list[ShellSpec]],
) -> list[BasisFunction]:
    functions: list[BasisFunction] = []
    for atom in atoms:
        if atom.symbol not in parsed_basis:
            raise ValueError(f"No basis found for element {atom.symbol}")
        for shell in parsed_basis[atom.symbol]:
            if shell.shell_type == "S":
                functions.append(make_basis_function(atom.coord, (0, 0, 0), shell))
            elif shell.shell_type == "P":
                functions.extend(
                    [
                        make_basis_function(atom.coord, (1, 0, 0), shell),
                        make_basis_function(atom.coord, (0, 1, 0), shell),
                        make_basis_function(atom.coord, (0, 0, 1), shell),
                    ]
                )
            else:
                raise ValueError(f"Unsupported shell type: {shell.shell_type}")
    return functions


def make_basis_function(
    center: np.ndarray,
    ang: tuple[int, int, int],
    shell: ShellSpec,
) -> BasisFunction:
    exponents = np.asarray(shell.exponents, dtype=float)
    raw_coeffs = np.asarray(shell.coeffs, dtype=float)
    primitive_norms = np.asarray(
        [primitive_normalization(alpha, ang) for alpha in exponents],
        dtype=float,
    )
    coeffs = raw_coeffs * primitive_norms

    norm = 0.0
    for i, alpha in enumerate(exponents):
        for j, beta in enumerate(exponents):
            norm += coeffs[i] * coeffs[j] * primitive_overlap(alpha, ang, center, beta, ang, center)
    coeffs /= math.sqrt(norm)

    return BasisFunction(center=center, ang=ang, exponents=exponents, coeffs=coeffs)


def primitive_normalization(alpha: float, ang: tuple[int, int, int]) -> float:
    l, m, n = ang
    denom = (
        double_factorial(2 * l - 1)
        * double_factorial(2 * m - 1)
        * double_factorial(2 * n - 1)
    )
    return (2.0 * alpha / math.pi) ** 0.75 * math.sqrt((4.0 * alpha) ** (l + m + n) / denom)


def double_factorial(value: int) -> int:
    if value <= 0:
        return 1
    result = 1
    for item in range(value, 0, -2):
        result *= item
    return result


@lru_cache(maxsize=None)
def hermite_coefficient(
    i: int,
    j: int,
    t: int,
    qx: float,
    alpha: float,
    beta: float,
) -> float:
    if t < 0 or t > i + j:
        return 0.0
    p = alpha + beta
    q = alpha * beta / p
    if i == 0 and j == 0:
        if t == 0:
            return math.exp(-q * qx * qx)
        return 0.0
    if j == 0:
        return (
            hermite_coefficient(i - 1, j, t - 1, qx, alpha, beta) / (2.0 * p)
            - q * qx * hermite_coefficient(i - 1, j, t, qx, alpha, beta) / alpha
            + (t + 1) * hermite_coefficient(i - 1, j, t + 1, qx, alpha, beta)
        )
    return (
        hermite_coefficient(i, j - 1, t - 1, qx, alpha, beta) / (2.0 * p)
        + q * qx * hermite_coefficient(i, j - 1, t, qx, alpha, beta) / beta
        + (t + 1) * hermite_coefficient(i, j - 1, t + 1, qx, alpha, beta)
    )


def overlap_1d(i: int, j: int, ax: float, bx: float, alpha: float, beta: float) -> float:
    if i < 0 or j < 0:
        return 0.0
    p = alpha + beta
    return hermite_coefficient(i, j, 0, ax - bx, alpha, beta) * math.sqrt(math.pi / p)


def primitive_overlap(
    alpha: float,
    ang_a: tuple[int, int, int],
    center_a: np.ndarray,
    beta: float,
    ang_b: tuple[int, int, int],
    center_b: np.ndarray,
) -> float:
    p = alpha + beta
    ex = hermite_coefficient(ang_a[0], ang_b[0], 0, center_a[0] - center_b[0], alpha, beta)
    ey = hermite_coefficient(ang_a[1], ang_b[1], 0, center_a[1] - center_b[1], alpha, beta)
    ez = hermite_coefficient(ang_a[2], ang_b[2], 0, center_a[2] - center_b[2], alpha, beta)
    return ex * ey * ez * (math.pi / p) ** 1.5


def primitive_kinetic(
    alpha: float,
    ang_a: tuple[int, int, int],
    center_a: np.ndarray,
    beta: float,
    ang_b: tuple[int, int, int],
    center_b: np.ndarray,
) -> float:
    sx = overlap_1d(ang_a[0], ang_b[0], center_a[0], center_b[0], alpha, beta)
    sy = overlap_1d(ang_a[1], ang_b[1], center_a[1], center_b[1], alpha, beta)
    sz = overlap_1d(ang_a[2], ang_b[2], center_a[2], center_b[2], alpha, beta)

    tx = kinetic_1d(ang_a[0], ang_b[0], center_a[0], center_b[0], alpha, beta)
    ty = kinetic_1d(ang_a[1], ang_b[1], center_a[1], center_b[1], alpha, beta)
    tz = kinetic_1d(ang_a[2], ang_b[2], center_a[2], center_b[2], alpha, beta)

    return tx * sy * sz + sx * ty * sz + sx * sy * tz


def kinetic_1d(i: int, j: int, ax: float, bx: float, alpha: float, beta: float) -> float:
    return (
        beta * (2 * j + 1) * overlap_1d(i, j, ax, bx, alpha, beta)
        - 2.0 * beta * beta * overlap_1d(i, j + 2, ax, bx, alpha, beta)
        - 0.5 * j * (j - 1) * overlap_1d(i, j - 2, ax, bx, alpha, beta)
    )


def gaussian_product_center(
    alpha: float,
    center_a: np.ndarray,
    beta: float,
    center_b: np.ndarray,
) -> np.ndarray:
    return (alpha * center_a + beta * center_b) / (alpha + beta)


def boys(n: int, x: float) -> float:
    if x < 1.0e-10:
        return 1.0 / (2 * n + 1) - x / (2 * n + 3)
    value = 0.5 * math.sqrt(math.pi / x) * math.erf(math.sqrt(x))
    if n == 0:
        return value
    exp_term = math.exp(-x)
    for m in range(n):
        value = ((2 * m + 1) * value - exp_term) / (2.0 * x)
    return value


@lru_cache(maxsize=None)
def coulomb_auxiliary(
    t: int,
    u: int,
    v: int,
    n: int,
    p: float,
    pcx: float,
    pcy: float,
    pcz: float,
    rpc2: float,
) -> float:
    if t == 0 and u == 0 and v == 0:
        return (-2.0 * p) ** n * boys(n, p * rpc2)
    if t > 0:
        value = pcx * coulomb_auxiliary(t - 1, u, v, n + 1, p, pcx, pcy, pcz, rpc2)
        if t > 1:
            value += (t - 1) * coulomb_auxiliary(t - 2, u, v, n + 1, p, pcx, pcy, pcz, rpc2)
        return value
    if u > 0:
        value = pcy * coulomb_auxiliary(t, u - 1, v, n + 1, p, pcx, pcy, pcz, rpc2)
        if u > 1:
            value += (u - 1) * coulomb_auxiliary(t, u - 2, v, n + 1, p, pcx, pcy, pcz, rpc2)
        return value
    value = pcz * coulomb_auxiliary(t, u, v - 1, n + 1, p, pcx, pcy, pcz, rpc2)
    if v > 1:
        value += (v - 1) * coulomb_auxiliary(t, u, v - 2, n + 1, p, pcx, pcy, pcz, rpc2)
    return value


def primitive_nuclear_attraction(
    alpha: float,
    ang_a: tuple[int, int, int],
    center_a: np.ndarray,
    beta: float,
    ang_b: tuple[int, int, int],
    center_b: np.ndarray,
    center_c: np.ndarray,
    charge_c: int,
) -> float:
    p = alpha + beta
    product_center = gaussian_product_center(alpha, center_a, beta, center_b)
    pc = product_center - center_c
    rpc2 = float(np.dot(pc, pc))

    value = 0.0
    for t in range(ang_a[0] + ang_b[0] + 1):
        ex = hermite_coefficient(ang_a[0], ang_b[0], t, center_a[0] - center_b[0], alpha, beta)
        for u in range(ang_a[1] + ang_b[1] + 1):
            ey = hermite_coefficient(ang_a[1], ang_b[1], u, center_a[1] - center_b[1], alpha, beta)
            for v in range(ang_a[2] + ang_b[2] + 1):
                ez = hermite_coefficient(ang_a[2], ang_b[2], v, center_a[2] - center_b[2], alpha, beta)
                value += ex * ey * ez * coulomb_auxiliary(
                    t,
                    u,
                    v,
                    0,
                    p,
                    pc[0],
                    pc[1],
                    pc[2],
                    rpc2,
                )
    return -charge_c * 2.0 * math.pi * value / p


def primitive_eri(
    alpha: float,
    ang_a: tuple[int, int, int],
    center_a: np.ndarray,
    beta: float,
    ang_b: tuple[int, int, int],
    center_b: np.ndarray,
    gamma: float,
    ang_c: tuple[int, int, int],
    center_c: np.ndarray,
    delta: float,
    ang_d: tuple[int, int, int],
    center_d: np.ndarray,
) -> float:
    p = alpha + beta
    q = gamma + delta
    product_ab = gaussian_product_center(alpha, center_a, beta, center_b)
    product_cd = gaussian_product_center(gamma, center_c, delta, center_d)
    pq = product_ab - product_cd
    rpq2 = float(np.dot(pq, pq))
    alpha_eri = p * q / (p + q)

    value = 0.0
    for t in range(ang_a[0] + ang_b[0] + 1):
        ex_ab = hermite_coefficient(ang_a[0], ang_b[0], t, center_a[0] - center_b[0], alpha, beta)
        for u in range(ang_a[1] + ang_b[1] + 1):
            ey_ab = hermite_coefficient(ang_a[1], ang_b[1], u, center_a[1] - center_b[1], alpha, beta)
            for v in range(ang_a[2] + ang_b[2] + 1):
                ez_ab = hermite_coefficient(ang_a[2], ang_b[2], v, center_a[2] - center_b[2], alpha, beta)
                for tau in range(ang_c[0] + ang_d[0] + 1):
                    ex_cd = hermite_coefficient(ang_c[0], ang_d[0], tau, center_c[0] - center_d[0], gamma, delta)
                    for nu in range(ang_c[1] + ang_d[1] + 1):
                        ey_cd = hermite_coefficient(ang_c[1], ang_d[1], nu, center_c[1] - center_d[1], gamma, delta)
                        for phi in range(ang_c[2] + ang_d[2] + 1):
                            ez_cd = hermite_coefficient(
                                ang_c[2],
                                ang_d[2],
                                phi,
                                center_c[2] - center_d[2],
                                gamma,
                                delta,
                            )
                            value += (
                                ex_ab
                                * ey_ab
                                * ez_ab
                                * ex_cd
                                * ey_cd
                                * ez_cd
                                * ((-1) ** (tau + nu + phi))
                                * coulomb_auxiliary(
                                    t + tau,
                                    u + nu,
                                    v + phi,
                                    0,
                                    alpha_eri,
                                    pq[0],
                                    pq[1],
                                    pq[2],
                                    rpq2,
                                )
                            )
    prefactor = 2.0 * math.pi**2.5 / (p * q * math.sqrt(p + q))
    return prefactor * value


def contracted_pair_integrals(
    bf_a: BasisFunction,
    bf_b: BasisFunction,
    atoms: list[Atom],
) -> tuple[float, float, float]:
    overlap = 0.0
    kinetic = 0.0
    nuclear = 0.0

    for ia, alpha in enumerate(bf_a.exponents):
        ca = bf_a.coeffs[ia]
        for ib, beta in enumerate(bf_b.exponents):
            cb = bf_b.coeffs[ib]
            prefactor = ca * cb
            overlap += prefactor * primitive_overlap(alpha, bf_a.ang, bf_a.center, beta, bf_b.ang, bf_b.center)
            kinetic += prefactor * primitive_kinetic(alpha, bf_a.ang, bf_a.center, beta, bf_b.ang, bf_b.center)
            for atom in atoms:
                nuclear += prefactor * primitive_nuclear_attraction(
                    alpha,
                    bf_a.ang,
                    bf_a.center,
                    beta,
                    bf_b.ang,
                    bf_b.center,
                    atom.coord,
                    atom.charge,
                )

    return overlap, kinetic, nuclear


def contracted_eri(
    bf_a: BasisFunction,
    bf_b: BasisFunction,
    bf_c: BasisFunction,
    bf_d: BasisFunction,
) -> float:
    value = 0.0
    for ia, alpha in enumerate(bf_a.exponents):
        ca = bf_a.coeffs[ia]
        for ib, beta in enumerate(bf_b.exponents):
            cb = bf_b.coeffs[ib]
            for ic, gamma in enumerate(bf_c.exponents):
                cc = bf_c.coeffs[ic]
                for idelta, delta in enumerate(bf_d.exponents):
                    cd = bf_d.coeffs[idelta]
                    value += ca * cb * cc * cd * primitive_eri(
                        alpha,
                        bf_a.ang,
                        bf_a.center,
                        beta,
                        bf_b.ang,
                        bf_b.center,
                        gamma,
                        bf_c.ang,
                        bf_c.center,
                        delta,
                        bf_d.ang,
                        bf_d.center,
                    )
    return value


def build_integrals(
    atoms: list[Atom],
    basis_functions: list[BasisFunction],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nbf = len(basis_functions)
    overlap = np.zeros((nbf, nbf), dtype=float)
    core_h = np.zeros((nbf, nbf), dtype=float)
    eri = np.zeros((nbf, nbf, nbf, nbf), dtype=float)

    for i in range(nbf):
        for j in range(i + 1):
            s_ij, t_ij, v_ij = contracted_pair_integrals(basis_functions[i], basis_functions[j], atoms)
            overlap[i, j] = overlap[j, i] = s_ij
            core_h[i, j] = core_h[j, i] = t_ij + v_ij

    for i in range(nbf):
        for j in range(i + 1):
            ij = i * (i + 1) // 2 + j
            for k in range(nbf):
                for l in range(k + 1):
                    kl = k * (k + 1) // 2 + l
                    if ij < kl:
                        continue
                    value = contracted_eri(
                        basis_functions[i],
                        basis_functions[j],
                        basis_functions[k],
                        basis_functions[l],
                    )
                    eri[i, j, k, l] = value
                    eri[j, i, k, l] = value
                    eri[i, j, l, k] = value
                    eri[j, i, l, k] = value
                    eri[k, l, i, j] = value
                    eri[l, k, i, j] = value
                    eri[k, l, j, i] = value
                    eri[l, k, j, i] = value

    return overlap, core_h, eri


def nuclear_repulsion_energy(atoms: list[Atom]) -> float:
    energy = 0.0
    for i, atom_i in enumerate(atoms):
        for atom_j in atoms[:i]:
            distance = float(np.linalg.norm(atom_i.coord - atom_j.coord))
            energy += atom_i.charge * atom_j.charge / distance
    return energy


def symmetric_orthogonalizer(overlap: cp.ndarray) -> cp.ndarray:
    eigvals, eigvecs = cp.linalg.eigh(overlap)
    if float(cp.min(eigvals).get()) < 1.0e-10:
        raise ValueError("Overlap matrix is singular or ill-conditioned")
    inv_sqrt = cp.diag(eigvals ** -0.5)
    return eigvecs @ inv_sqrt @ eigvecs.T


def rhf_scf(
    core_h: np.ndarray,
    overlap: np.ndarray,
    eri: np.ndarray,
    nelec: int,
    e_nuc: float,
    max_cycle: int,
    conv_tol: float,
    density_tol: float,
    verbose: bool,
) -> dict[str, object]:
    nocc = nelec // 2
    h_gpu = cp.asarray(core_h, dtype=cp.float64)
    s_gpu = cp.asarray(overlap, dtype=cp.float64)
    eri_gpu = cp.asarray(eri, dtype=cp.float64)
    x_gpu = symmetric_orthogonalizer(s_gpu)

    density = cp.zeros_like(h_gpu)
    energy_prev: float | None = None
    history: list[dict[str, float]] = []
    converged = False
    orbital_energies = cp.array([])

    for cycle in range(1, max_cycle + 1):
        j_mat = cp.einsum("pqrs,rs->pq", eri_gpu, density)
        k_mat = cp.einsum("prqs,rs->pq", eri_gpu, density)
        fock = h_gpu + j_mat - 0.5 * k_mat

        fock_ortho = x_gpu.T @ fock @ x_gpu
        orbital_energies, coeffs_ortho = cp.linalg.eigh(fock_ortho)
        coeffs = x_gpu @ coeffs_ortho
        coeffs_occ = coeffs[:, :nocc]
        density_new = 2.0 * (coeffs_occ @ coeffs_occ.T)

        j_new = cp.einsum("pqrs,rs->pq", eri_gpu, density_new)
        k_new = cp.einsum("prqs,rs->pq", eri_gpu, density_new)
        fock_new = h_gpu + j_new - 0.5 * k_new

        e_elec = 0.5 * cp.sum(density_new * (h_gpu + fock_new))
        total_energy = float(e_elec.get()) + e_nuc
        delta_e = math.inf if energy_prev is None else abs(total_energy - energy_prev)
        rms_density = float(cp.sqrt(cp.mean((density_new - density) ** 2)).get())

        history.append(
            {
                "cycle": float(cycle),
                "energy_hartree": total_energy,
                "delta_e": delta_e,
                "rms_density": rms_density,
            }
        )
        if verbose:
            print(
                f"cycle={cycle:02d} "
                f"E={total_energy:.12f} "
                f"dE={delta_e:.3e} "
                f"rmsD={rms_density:.3e}"
            )

        density = density_new
        energy_prev = total_energy
        if delta_e < conv_tol and rms_density < density_tol:
            converged = True
            break

    return {
        "converged": converged,
        "cycles": len(history),
        "history": history,
        "energy_hartree": energy_prev,
        "orbital_energies_hartree": cp.asnumpy(orbital_energies).tolist(),
    }


def main() -> None:
    args = parse_args()
    parsed_basis = parse_orca_basis(args.basis)
    geometry = get_geometry(args)
    atoms, nelec = build_atoms(geometry, args.charge)
    basis_functions = build_basis_functions(atoms, parsed_basis)

    overlap, core_h, eri = build_integrals(atoms, basis_functions)
    e_nuc = nuclear_repulsion_energy(atoms)
    scf = rhf_scf(
        core_h=core_h,
        overlap=overlap,
        eri=eri,
        nelec=nelec,
        e_nuc=e_nuc,
        max_cycle=args.max_cycle,
        conv_tol=args.conv_tol,
        density_tol=args.density_tol,
        verbose=args.verbose,
    )

    summary = {
        "basis": str(args.basis),
        "charge": args.charge,
        "converged": scf["converged"],
        "cycles": scf["cycles"],
        "electron_count": nelec,
        "energy_hartree": scf["energy_hartree"],
        "nuclear_repulsion_hartree": e_nuc,
        "num_atoms": len(atoms),
        "num_basis_functions": len(basis_functions),
        "orbital_energies_hartree": scf["orbital_energies_hartree"],
        "scf_history": scf["history"],
        "system": [
            {
                "symbol": atom.symbol,
                "coord_bohr": atom.coord.tolist(),
            }
            for atom in atoms
        ],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
