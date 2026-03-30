from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cupy as cp
import numpy as np

from b3lyp_xc import B3LYP_EXACT_EXCHANGE, evaluate_b3lyp_xc, get_b3lyp_backend


ANGSTROM_TO_BOHR = 1.8897259886
ATOM_NAMES = {
    "H": "H",
    "HYDROGEN": "H",
    "C": "C",
    "CARBON": "C",
    "O": "O",
    "OXYGEN": "O",
}
ATOMIC_NUMBERS = {
    "H": 1,
    "C": 6,
    "O": 8,
}
GRID_SCALES_BOHR = {
    "H": 1.5,
    "C": 2.2,
    "O": 2.0,
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


@dataclass(frozen=True)
class Shell:
    center: np.ndarray
    ang_momentum: int
    basis_indices: tuple[int, ...]


class DIIS:
    def __init__(self, max_vectors: int = 6) -> None:
        self.max_vectors = max_vectors
        self._focks: list[np.ndarray] = []
        self._errors: list[np.ndarray] = []

    def extrapolate(
        self,
        fock: cp.ndarray,
        density: cp.ndarray,
        overlap: cp.ndarray,
    ) -> cp.ndarray:
        error = fock @ density @ overlap - overlap @ density @ fock
        self._focks.append(cp.asnumpy(fock))
        self._errors.append(cp.asnumpy(error).ravel())
        if len(self._focks) > self.max_vectors:
            self._focks.pop(0)
            self._errors.pop(0)
        if len(self._focks) < 2:
            return fock

        size = len(self._focks)
        b_mat = np.empty((size + 1, size + 1), dtype=float)
        b_mat[-1, :] = -1.0
        b_mat[:, -1] = -1.0
        b_mat[-1, -1] = 0.0
        for i, err_i in enumerate(self._errors):
            for j, err_j in enumerate(self._errors[: i + 1]):
                value = float(np.dot(err_i, err_j))
                b_mat[i, j] = value
                b_mat[j, i] = value

        rhs = np.zeros(size + 1, dtype=float)
        rhs[-1] = -1.0
        try:
            coeffs = np.linalg.solve(b_mat, rhs)[:-1]
        except np.linalg.LinAlgError:
            return fock

        mixed = sum(weight * mat for weight, mat in zip(coeffs, self._focks, strict=True))
        return cp.asarray(mixed, dtype=cp.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Self-contained restricted HF / B3LYP SCF using analytic Gaussian "
            "integrals, a CuPy-backed SCF loop, and an exact PySCF/libxc B3LYP "
            "semilocal XC backend with a baked fallback."
        )
    )
    parser.add_argument(
        "--basis",
        type=Path,
        default=Path(__file__).with_name("def2-TZVP.orca"),
        help="Path to an ORCA-format basis file.",
    )
    parser.add_argument(
        "--method",
        choices=("rhf", "b3lyp"),
        default="b3lyp",
        help="Electronic structure method to run.",
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
    parser.add_argument("--conv-tol", type=float, default=1e-8)
    parser.add_argument("--density-tol", type=float, default=1e-6)
    parser.add_argument("--grid-radial", type=int, default=32)
    parser.add_argument("--grid-angular", type=int, default=86)
    parser.add_argument(
        "--partition-power",
        type=float,
        default=4.0,
        help="Inverse-distance exponent for multicenter grid partitioning.",
    )
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
        if head not in {"S", "P", "D", "F", "L"}:
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

        if head == "L":
            basis[current_atom].append(
                ShellSpec("S", tuple(exponents), tuple(coeffs))
            )
            basis[current_atom].append(
                ShellSpec("P", tuple(exponents), tuple(coeffs_p or ()))
            )
            continue

        basis[current_atom].append(
            ShellSpec(head, tuple(exponents), tuple(coeffs))
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
        raise ValueError("This implementation only supports closed-shell systems")
    return atoms, nelec


@lru_cache(maxsize=None)
def cartesian_tuples(total_ang_momentum: int) -> tuple[tuple[int, int, int], ...]:
    terms: list[tuple[int, int, int]] = []
    for lx in range(total_ang_momentum, -1, -1):
        for ly in range(total_ang_momentum - lx, -1, -1):
            lz = total_ang_momentum - lx - ly
            terms.append((lx, ly, lz))
    return tuple(terms)


def build_shells_and_basis_functions(
    atoms: list[Atom],
    parsed_basis: dict[str, list[ShellSpec]],
) -> tuple[list[Shell], list[BasisFunction]]:
    shell_to_l = {"S": 0, "P": 1, "D": 2, "F": 3}
    shells: list[Shell] = []
    functions: list[BasisFunction] = []
    for atom in atoms:
        if atom.symbol not in parsed_basis:
            raise ValueError(f"No basis found for element {atom.symbol}")
        for shell in parsed_basis[atom.symbol]:
            ang_momentum = shell_to_l[shell.shell_type]
            start = len(functions)
            for ang in cartesian_tuples(ang_momentum):
                functions.append(make_basis_function(atom.coord, ang, shell))
            stop = len(functions)
            shells.append(
                Shell(
                    center=atom.coord,
                    ang_momentum=ang_momentum,
                    basis_indices=tuple(range(start, stop)),
                )
            )
    return shells, functions


def build_basis_functions(
    atoms: list[Atom],
    parsed_basis: dict[str, list[ShellSpec]],
) -> list[BasisFunction]:
    _, functions = build_shells_and_basis_functions(atoms, parsed_basis)
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
            return math.exp(-q * qx ** 2)
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
                    ex_cd = hermite_coefficient(
                        ang_c[0], ang_d[0], tau, center_c[0] - center_d[0], gamma, delta
                    )
                    for nu in range(ang_c[1] + ang_d[1] + 1):
                        ey_cd = hermite_coefficient(
                            ang_c[1], ang_d[1], nu, center_c[1] - center_d[1], gamma, delta
                        )
                        for phi in range(ang_c[2] + ang_d[2] + 1):
                            ez_cd = hermite_coefficient(
                                ang_c[2], ang_d[2], phi, center_c[2] - center_d[2], gamma, delta
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


@lru_cache(maxsize=None)
def hermite_tensor_indices(max_axis_ang_momentum: int) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (t, u, v)
        for t in range(max_axis_ang_momentum + 1)
        for u in range(max_axis_ang_momentum + 1)
        for v in range(max_axis_ang_momentum + 1)
    )


def shell_basis_functions(
    shell: Shell,
    basis_functions: list[BasisFunction],
) -> list[BasisFunction]:
    return [basis_functions[index] for index in shell.basis_indices]


def primitive_shell_pair_transform(
    shell_a: Shell,
    shell_b: Shell,
    basis_functions: list[BasisFunction],
    primitive_a: int,
    primitive_b: int,
    alpha: float,
    beta: float,
) -> tuple[np.ndarray, tuple[tuple[int, int, int], ...]]:
    basis_a = shell_basis_functions(shell_a, basis_functions)
    basis_b = shell_basis_functions(shell_b, basis_functions)
    terms = hermite_tensor_indices(shell_a.ang_momentum + shell_b.ang_momentum)
    transform = np.zeros((len(basis_a) * len(basis_b), len(terms)), dtype=float)

    ab = shell_a.center - shell_b.center
    for row_a, bf_a in enumerate(basis_a):
        coeff_a = bf_a.coeffs[primitive_a]
        ax, ay, az = bf_a.ang
        for row_b, bf_b in enumerate(basis_b):
            coeff_b = bf_b.coeffs[primitive_b]
            bx, by, bz = bf_b.ang
            row = row_a * len(basis_b) + row_b
            scale = coeff_a * coeff_b
            for col, (t, u, v) in enumerate(terms):
                transform[row, col] = scale * (
                    hermite_coefficient(ax, bx, t, ab[0], alpha, beta)
                    * hermite_coefficient(ay, by, u, ab[1], alpha, beta)
                    * hermite_coefficient(az, bz, v, ab[2], alpha, beta)
                )

    return transform, terms


def compute_r_integrals(
    max_t: int,
    max_u: int,
    max_v: int,
    alpha_eri: float,
    pq: np.ndarray,
    rpq2: float,
) -> np.ndarray:
    r = np.empty((max_t + 1, max_u + 1, max_v + 1), dtype=float)
    for t in range(max_t + 1):
        for u in range(max_u + 1):
            for v in range(max_v + 1):
                r[t, u, v] = coulomb_auxiliary(
                    t,
                    u,
                    v,
                    0,
                    alpha_eri,
                    float(pq[0]),
                    float(pq[1]),
                    float(pq[2]),
                    rpq2,
                )
    return r


def compute_pq_integrals(
    ab_terms: tuple[tuple[int, int, int], ...],
    cd_terms: tuple[tuple[int, int, int], ...],
    r_integrals: np.ndarray,
) -> np.ndarray:
    pq_integrals = np.empty((len(ab_terms), len(cd_terms)), dtype=float)
    for row, (t, u, v) in enumerate(ab_terms):
        for col, (tau, nu, phi) in enumerate(cd_terms):
            pq_integrals[row, col] = ((-1) ** (tau + nu + phi)) * r_integrals[
                t + tau,
                u + nu,
                v + phi,
            ]
    return pq_integrals


def shell_quartet_block(
    shell_a: Shell,
    shell_b: Shell,
    shell_c: Shell,
    shell_d: Shell,
    basis_functions: list[BasisFunction],
) -> np.ndarray:
    basis_a = shell_basis_functions(shell_a, basis_functions)
    basis_b = shell_basis_functions(shell_b, basis_functions)
    basis_c = shell_basis_functions(shell_c, basis_functions)
    basis_d = shell_basis_functions(shell_d, basis_functions)

    ref_a = basis_a[0]
    ref_b = basis_b[0]
    ref_c = basis_c[0]
    ref_d = basis_d[0]

    block = np.zeros((len(basis_a) * len(basis_b), len(basis_c) * len(basis_d)), dtype=float)
    product_ab_terms: dict[tuple[int, int], tuple[np.ndarray, tuple[tuple[int, int, int], ...], float, np.ndarray]] = {}

    for primitive_a, alpha in enumerate(ref_a.exponents):
        for primitive_b, beta in enumerate(ref_b.exponents):
            e_ab, ab_terms = primitive_shell_pair_transform(
                shell_a,
                shell_b,
                basis_functions,
                primitive_a,
                primitive_b,
                alpha,
                beta,
            )
            p = alpha + beta
            product_ab_terms[(primitive_a, primitive_b)] = (
                e_ab,
                ab_terms,
                p,
                gaussian_product_center(alpha, shell_a.center, beta, shell_b.center),
            )

    for primitive_a, primitive_b in product_ab_terms:
        e_ab, ab_terms, p, product_ab = product_ab_terms[(primitive_a, primitive_b)]
        alpha = ref_a.exponents[primitive_a]
        beta = ref_b.exponents[primitive_b]

        for primitive_c, gamma in enumerate(ref_c.exponents):
            for primitive_d, delta in enumerate(ref_d.exponents):
                e_cd, cd_terms = primitive_shell_pair_transform(
                    shell_c,
                    shell_d,
                    basis_functions,
                    primitive_c,
                    primitive_d,
                    gamma,
                    delta,
                )
                q = gamma + delta
                product_cd = gaussian_product_center(gamma, shell_c.center, delta, shell_d.center)
                pq = product_ab - product_cd
                rpq2 = float(np.dot(pq, pq))
                alpha_eri = p * q / (p + q)

                r_integrals = compute_r_integrals(
                    shell_a.ang_momentum + shell_b.ang_momentum + shell_c.ang_momentum + shell_d.ang_momentum,
                    shell_a.ang_momentum + shell_b.ang_momentum + shell_c.ang_momentum + shell_d.ang_momentum,
                    shell_a.ang_momentum + shell_b.ang_momentum + shell_c.ang_momentum + shell_d.ang_momentum,
                    alpha_eri,
                    pq,
                    rpq2,
                )
                pq_integrals = compute_pq_integrals(ab_terms, cd_terms, r_integrals)
                prefactor = 2.0 * math.pi**2.5 / (p * q * math.sqrt(p + q))
                block += prefactor * (e_ab @ pq_integrals @ e_cd.T)

    return block.reshape(len(basis_a), len(basis_b), len(basis_c), len(basis_d))


def build_integrals(
    atoms: list[Atom],
    basis_functions: list[BasisFunction],
    shells: list[Shell],
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

    for shell_i, shell_a in enumerate(shells):
        a_idx = shell_a.basis_indices
        for shell_j, shell_b in enumerate(shells[: shell_i + 1]):
            b_idx = shell_b.basis_indices
            ij = shell_i * (shell_i + 1) // 2 + shell_j
            for shell_k, shell_c in enumerate(shells):
                c_idx = shell_c.basis_indices
                for shell_l, shell_d in enumerate(shells[: shell_k + 1]):
                    d_idx = shell_d.basis_indices
                    kl = shell_k * (shell_k + 1) // 2 + shell_l
                    if ij < kl:
                        continue

                    block = shell_quartet_block(shell_a, shell_b, shell_c, shell_d, basis_functions)
                    for local_a, global_a in enumerate(a_idx):
                        for local_b, global_b in enumerate(b_idx):
                            for local_c, global_c in enumerate(c_idx):
                                for local_d, global_d in enumerate(d_idx):
                                    value = block[local_a, local_b, local_c, local_d]
                                    eri[global_a, global_b, global_c, global_d] = value
                                    eri[global_b, global_a, global_c, global_d] = value
                                    eri[global_a, global_b, global_d, global_c] = value
                                    eri[global_b, global_a, global_d, global_c] = value
                                    eri[global_c, global_d, global_a, global_b] = value
                                    eri[global_d, global_c, global_a, global_b] = value
                                    eri[global_c, global_d, global_b, global_a] = value
                                    eri[global_d, global_c, global_b, global_a] = value

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


def fibonacci_sphere(num_points: int) -> tuple[np.ndarray, np.ndarray]:
    if num_points < 6:
        raise ValueError("--grid-angular must be at least 6")
    idx = np.arange(num_points, dtype=float)
    z = 1.0 - 2.0 * (idx + 0.5) / num_points
    phi = math.pi * (3.0 - math.sqrt(5.0)) * idx
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    points = np.stack([radius * np.cos(phi), radius * np.sin(phi), z], axis=1)
    weights = np.full(num_points, 4.0 * math.pi / num_points, dtype=float)
    return points, weights


def radial_grid(num_points: int, scale: float) -> tuple[np.ndarray, np.ndarray]:
    if num_points < 8:
        raise ValueError("--grid-radial must be at least 8")
    nodes, weights = np.polynomial.legendre.leggauss(num_points)
    t = 0.5 * (nodes + 1.0)
    radial = scale * t / (1.0 - t)
    jacobian = 0.5 * scale / (1.0 - t) ** 2
    radial_weights = weights * jacobian * radial * radial
    return radial, radial_weights


def build_grid(
    atoms: list[Atom],
    num_radial: int,
    num_angular: int,
    partition_power: float,
) -> tuple[np.ndarray, np.ndarray]:
    angular_points, angular_weights = fibonacci_sphere(num_angular)
    centers = np.asarray([atom.coord for atom in atoms], dtype=float)

    all_points: list[np.ndarray] = []
    all_weights: list[np.ndarray] = []
    for atom_index, atom in enumerate(atoms):
        scale = GRID_SCALES_BOHR.get(atom.symbol, 2.0)
        radial_points, radial_weights = radial_grid(num_radial, scale)
        shell_points = atom.coord + radial_points[:, None, None] * angular_points[None, :, :]
        shell_weights = radial_weights[:, None] * angular_weights[None, :]

        points = shell_points.reshape(-1, 3)
        weights = shell_weights.reshape(-1)

        distances = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        raw = 1.0 / np.maximum(distances, 1.0e-10) ** partition_power
        partition = raw[:, atom_index] / np.sum(raw, axis=1)

        all_points.append(points)
        all_weights.append(weights * partition)

    points = np.concatenate(all_points, axis=0)
    weights = np.concatenate(all_weights, axis=0)
    mask = weights > 1.0e-16
    return points[mask], weights[mask]


def power_component(values: cp.ndarray, exponent: int) -> cp.ndarray:
    if exponent == 0:
        return cp.ones_like(values)
    return values ** exponent


def evaluate_basis_on_grid(
    basis_functions: list[BasisFunction],
    grid_points: np.ndarray,
) -> tuple[cp.ndarray, cp.ndarray]:
    points = cp.asarray(grid_points, dtype=cp.float64)
    ngrid = points.shape[0]
    nbf = len(basis_functions)
    values = cp.empty((ngrid, nbf), dtype=cp.float64)
    gradients = cp.empty((3, ngrid, nbf), dtype=cp.float64)

    for ibf, bf in enumerate(basis_functions):
        center = cp.asarray(bf.center, dtype=cp.float64)
        exponents = cp.asarray(bf.exponents, dtype=cp.float64)
        coeffs = cp.asarray(bf.coeffs, dtype=cp.float64)

        diff = points - center[None, :]
        dx = diff[:, 0]
        dy = diff[:, 1]
        dz = diff[:, 2]
        r2 = cp.sum(diff * diff, axis=1)

        lx, ly, lz = bf.ang
        x_l = power_component(dx, lx)
        y_m = power_component(dy, ly)
        z_n = power_component(dz, lz)
        x_lm1 = power_component(dx, lx - 1) if lx > 0 else cp.ones_like(dx)
        y_mm1 = power_component(dy, ly - 1) if ly > 0 else cp.ones_like(dy)
        z_nm1 = power_component(dz, lz - 1) if lz > 0 else cp.ones_like(dz)

        phi = cp.zeros(ngrid, dtype=cp.float64)
        grad_x = cp.zeros_like(phi)
        grad_y = cp.zeros_like(phi)
        grad_z = cp.zeros_like(phi)
        for alpha, coeff in zip(exponents, coeffs, strict=True):
            exp_term = cp.exp(-alpha * r2)
            poly = x_l * y_m * z_n
            contrib = coeff * poly * exp_term
            phi += contrib

            dpoly_x = (lx * x_lm1 * y_m * z_n) if lx > 0 else 0.0
            dpoly_y = (ly * x_l * y_mm1 * z_n) if ly > 0 else 0.0
            dpoly_z = (lz * x_l * y_m * z_nm1) if lz > 0 else 0.0
            grad_x += coeff * exp_term * (dpoly_x - 2.0 * alpha * dx * poly)
            grad_y += coeff * exp_term * (dpoly_y - 2.0 * alpha * dy * poly)
            grad_z += coeff * exp_term * (dpoly_z - 2.0 * alpha * dz * poly)

        values[:, ibf] = phi
        gradients[0, :, ibf] = grad_x
        gradients[1, :, ibf] = grad_y
        gradients[2, :, ibf] = grad_z

    return values, gradients


def density_and_gradient(
    density: cp.ndarray,
    values: cp.ndarray,
    gradients: cp.ndarray,
) -> tuple[cp.ndarray, cp.ndarray]:
    tmp = values @ density
    rho = cp.sum(tmp * values, axis=1)

    grad_rho = cp.empty((3, values.shape[0]), dtype=cp.float64)
    for axis in range(3):
        grad_rho[axis] = 2.0 * cp.sum(tmp * gradients[axis], axis=1)
    return rho, grad_rho


def build_xc_matrix_and_energy(
    density: cp.ndarray,
    values: cp.ndarray,
    gradients: cp.ndarray,
    weights: cp.ndarray,
) -> tuple[cp.ndarray, float]:
    rho, grad_rho = density_and_gradient(density, values, gradients)
    grad_norm = cp.sqrt(cp.sum(grad_rho * grad_rho, axis=0))
    energy_density, vrho, vgrad = evaluate_b3lyp_xc(rho, grad_norm)

    vmat = values.T @ (weights[:, None] * vrho[:, None] * values)

    safe_grad_norm = cp.where(grad_norm > 1.0e-18, grad_norm, 1.0)
    direction = grad_rho / safe_grad_norm[None, :]
    for axis in range(3):
        factor = weights * vgrad * direction[axis]
        mixed = gradients[axis].T @ (factor[:, None] * values)
        vmat += mixed + mixed.T

    e_xc = float(cp.sum(weights * energy_density).get())
    return vmat, e_xc


def initial_density(
    core_h: cp.ndarray,
    orthogonalizer: cp.ndarray,
    nocc: int,
) -> cp.ndarray:
    fock_ortho = orthogonalizer.T @ core_h @ orthogonalizer
    _, coeffs_ortho = cp.linalg.eigh(fock_ortho)
    coeffs = orthogonalizer @ coeffs_ortho
    coeffs_occ = coeffs[:, :nocc]
    return 2.0 * (coeffs_occ @ coeffs_occ.T)


def restricted_scf(
    method: str,
    core_h: np.ndarray,
    overlap: np.ndarray,
    eri: np.ndarray,
    nelec: int,
    e_nuc: float,
    basis_functions: list[BasisFunction],
    grid_points: np.ndarray | None,
    grid_weights: np.ndarray | None,
    max_cycle: int,
    conv_tol: float,
    density_tol: float,
    verbose: bool,
) -> dict[str, object]:
    nocc = nelec // 2
    exact_exchange_fraction = 1.0 if method == "rhf" else B3LYP_EXACT_EXCHANGE

    h_gpu = cp.asarray(core_h, dtype=cp.float64)
    s_gpu = cp.asarray(overlap, dtype=cp.float64)
    eri_gpu = cp.asarray(eri, dtype=cp.float64)
    x_gpu = symmetric_orthogonalizer(s_gpu)
    density = initial_density(h_gpu, x_gpu, nocc)
    diis = DIIS()

    values = gradients = weights_gpu = None
    if method == "b3lyp":
        if grid_points is None or grid_weights is None:
            raise ValueError("B3LYP requires a numerical integration grid")
        values, gradients = evaluate_basis_on_grid(basis_functions, grid_points)
        weights_gpu = cp.asarray(grid_weights, dtype=cp.float64)

    energy_prev: float | None = None
    history: list[dict[str, float]] = []
    converged = False
    orbital_energies = cp.array([], dtype=cp.float64)

    for cycle in range(1, max_cycle + 1):
        j_mat = cp.einsum("pqrs,rs->pq", eri_gpu, density)
        k_mat = cp.einsum("prqs,rs->pq", eri_gpu, density)
        xc_matrix = cp.zeros_like(h_gpu)
        e_xc = 0.0
        if method == "b3lyp":
            xc_matrix, e_xc = build_xc_matrix_and_energy(density, values, gradients, weights_gpu)

        fock = h_gpu + j_mat - 0.5 * exact_exchange_fraction * k_mat + xc_matrix
        fock = diis.extrapolate(fock, density, s_gpu)

        fock_ortho = x_gpu.T @ fock @ x_gpu
        orbital_energies, coeffs_ortho = cp.linalg.eigh(fock_ortho)
        coeffs = x_gpu @ coeffs_ortho
        coeffs_occ = coeffs[:, :nocc]
        density_new = 2.0 * (coeffs_occ @ coeffs_occ.T)

        j_new = cp.einsum("pqrs,rs->pq", eri_gpu, density_new)
        k_new = cp.einsum("prqs,rs->pq", eri_gpu, density_new)

        e_xc_new = 0.0
        if method == "b3lyp":
            _, e_xc_new = build_xc_matrix_and_energy(density_new, values, gradients, weights_gpu)

        e_one = float(cp.sum(density_new * h_gpu).get())
        e_coulomb = 0.5 * float(cp.sum(density_new * j_new).get())
        e_exact_exchange = -0.25 * exact_exchange_fraction * float(cp.sum(density_new * k_new).get())
        total_energy = e_one + e_coulomb + e_exact_exchange + e_xc_new + e_nuc

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
    shells, basis_functions = build_shells_and_basis_functions(atoms, parsed_basis)

    if args.verbose:
        print(f"building one- and two-electron integrals for {len(basis_functions)} basis functions")
    overlap, core_h, eri = build_integrals(atoms, basis_functions, shells)
    e_nuc = nuclear_repulsion_energy(atoms)

    grid_points = grid_weights = None
    if args.method == "b3lyp":
        if args.verbose:
            print(f"using B3LYP XC backend: {get_b3lyp_backend()}")
        if args.verbose:
            print(
                f"building multicenter grid with {args.grid_radial} radial x "
                f"{args.grid_angular} angular points per atom"
            )
        grid_points, grid_weights = build_grid(
            atoms=atoms,
            num_radial=args.grid_radial,
            num_angular=args.grid_angular,
            partition_power=args.partition_power,
        )

    scf = restricted_scf(
        method=args.method,
        core_h=core_h,
        overlap=overlap,
        eri=eri,
        nelec=nelec,
        e_nuc=e_nuc,
        basis_functions=basis_functions,
        grid_points=grid_points,
        grid_weights=grid_weights,
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
        "grid_angular": args.grid_angular if grid_points is not None else None,
        "grid_points": int(len(grid_weights)) if grid_weights is not None else None,
        "grid_radial": args.grid_radial if grid_points is not None else None,
        "method": args.method,
        "nuclear_repulsion_hartree": e_nuc,
        "num_atoms": len(atoms),
        "num_basis_functions": len(basis_functions),
        "orbital_energies_hartree": scf["orbital_energies_hartree"],
        "scf_history": scf["history"],
        "xc_backend": get_b3lyp_backend() if args.method == "b3lyp" else None,
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
