import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from b3lyp_xc import B3LYP_EXACT_EXCHANGE, evaluate_b3lyp_xc, get_b3lyp_backend
from eri import batched_hermite_tensor, build_eri_tensor, compute_r_integrals_batched


ANGSTROM_TO_BOHR = 1.8897259886
ATOM_NAMES = {
    "H": "H", "HYDROGEN": "H",
    "C": "C", "CARBON": "C",
    "O": "O", "OXYGEN": "O",
}
ATOMIC_NUMBERS = {
    "H": 1, "C": 6, "O": 8,
}
GRID_SCALES_BOHR = {
    "H": 1.5, "C": 2.2, "O": 2.0,
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
        self._focks: list[torch.Tensor] = []
        self._errors: list[torch.Tensor] = []

    def extrapolate(
        self,
        fock: torch.Tensor,
        density: torch.Tensor,
        overlap: torch.Tensor,
    ) -> torch.Tensor:
        error = fock @ density @ overlap - overlap @ density @ fock
        self._focks.append(fock.detach().clone())
        self._errors.append(error.detach().clone().flatten())
        if len(self._focks) > self.max_vectors:
            self._focks.pop(0)
            self._errors.pop(0)
        if len(self._focks) < 2:
            return fock

        size = len(self._focks)
        b_mat = torch.empty((size + 1, size + 1), dtype=fock.dtype, device=fock.device)
        b_mat[-1, :] = -1.0
        b_mat[:, -1] = -1.0
        b_mat[-1, -1] = 0.0
        for i, err_i in enumerate(self._errors):
            for j, err_j in enumerate(self._errors[: i + 1]):
                value = torch.dot(err_i, err_j)
                b_mat[i, j] = value
                b_mat[j, i] = value

        rhs = torch.zeros(size + 1, dtype=fock.dtype, device=fock.device)
        rhs[-1] = -1.0
        try:
            coeffs = torch.linalg.solve(b_mat, rhs)[:-1]
        except RuntimeError:
            return fock

        mixed = sum(weight * mat for weight, mat in zip(coeffs, self._focks, strict=True))
        return mixed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Self-contained restricted HF / B3LYP SCF using fully batched PyTorch "
            "GPU tensor integrals and a PyTorch-backed SCF loop."
        )
    )
    parser.add_argument(
        "--basis", type=Path, default=Path(__file__).with_name("def2-TZVP.orca"),
        help="Path to an ORCA-format basis file."
    )
    parser.add_argument("--method", choices=("rhf", "b3lyp"), default="b3lyp")
    parser.add_argument("--xyz", type=Path)
    parser.add_argument("--system", choices=("water", "h2"), default="water")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--max-cycle", type=int, default=50)
    parser.add_argument("--conv-tol", type=float, default=1e-8)
    parser.add_argument("--density-tol", type=float, default=1e-6)
    parser.add_argument("--grid-radial", type=int, default=32)
    parser.add_argument("--grid-angular", type=int, default=86)
    parser.add_argument("--partition-power", type=float, default=4.0)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_xyz(path: Path) -> list[tuple[str, tuple[float, float, float]]]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not lines: raise ValueError(f"XYZ file is empty: {path}")
    try:
        natoms = int(lines[0])
        body = lines[2 : 2 + natoms]
    except ValueError:
        body = lines
    atoms: list[tuple[str, tuple[float, float, float]]] = []
    for line in body:
        fields = line.split()
        symbol = normalize_symbol(fields[0])
        coord = (float(fields[1]), float(fields[2]), float(fields[3]))
        atoms.append((symbol, coord))
    return atoms


def get_geometry(args: argparse.Namespace) -> list[tuple[str, tuple[float, float, float]]]:
    if args.xyz is not None: return load_xyz(args.xyz)
    if args.system == "h2": return list(H2)
    return list(WATER)


def normalize_symbol(label: str) -> str:
    key = label.strip().upper()
    if key not in ATOM_NAMES: raise ValueError(f"Unsupported atom label: {label}")
    return ATOM_NAMES[key]


def parse_orca_basis(path: Path) -> dict[str, list[ShellSpec]]:
    lines = path.read_text().splitlines()
    in_data, current_atom, basis, i = False, None, {}, 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("!"): continue
        if line == "$DATA":
            in_data = True; continue
        if line == "$END": break
        if not in_data: continue

        fields = line.split()
        head = fields[0].upper()
        if head in ATOM_NAMES:
            current_atom = normalize_symbol(head)
            basis.setdefault(current_atom, [])
            continue

        nprims = int(fields[1])
        exponents, coeffs = [], []
        coeffs_p: list[float] | None = [] if head == "L" else None
        for _ in range(nprims):
            row = lines[i].split()
            i += 1
            exponents.append(float(row[1]))
            coeffs.append(float(row[2]))
            if coeffs_p is not None: coeffs_p.append(float(row[3]))

        if head == "L":
            basis[current_atom].append(ShellSpec("S", tuple(exponents), tuple(coeffs)))
            basis[current_atom].append(ShellSpec("P", tuple(exponents), tuple(coeffs_p or ())))
            continue
        basis[current_atom].append(ShellSpec(head, tuple(exponents), tuple(coeffs)))
    return basis


def build_atoms(geometry: list[tuple[str, tuple[float, float, float]]], charge: int) -> tuple[list[Atom], int]:
    atoms, total_nuclear_charge = [], 0
    for symbol, coord_angstrom in geometry:
        coord_bohr = np.asarray(coord_angstrom, dtype=float) * ANGSTROM_TO_BOHR
        z = ATOMIC_NUMBERS[symbol]
        atoms.append(Atom(symbol=symbol, charge=z, coord=coord_bohr))
        total_nuclear_charge += z

    nelec = total_nuclear_charge - charge
    if nelec <= 0 or nelec % 2 != 0: raise ValueError("Invalid closed-shell properties.")
    return atoms, nelec


@lru_cache(maxsize=None)
def cartesian_tuples(total_ang_momentum: int) -> tuple[tuple[int, int, int], ...]:
    terms = []
    for lx in range(total_ang_momentum, -1, -1):
        for ly in range(total_ang_momentum - lx, -1, -1):
            terms.append((lx, ly, total_ang_momentum - lx - ly))
    return tuple(terms)


def build_shells_and_basis_functions(atoms: list[Atom], parsed_basis: dict[str, list[ShellSpec]]) -> tuple[list[Shell], list[BasisFunction]]:
    shell_to_l = {"S": 0, "P": 1, "D": 2, "F": 3}
    shells, functions = [], []
    for atom in atoms:
        for shell in parsed_basis[atom.symbol]:
            ang_momentum = shell_to_l[shell.shell_type]
            start = len(functions)
            for ang in cartesian_tuples(ang_momentum):
                functions.append(make_basis_function(atom.coord, ang, shell))
            shells.append(Shell(center=atom.coord, ang_momentum=ang_momentum, basis_indices=tuple(range(start, len(functions)))))
    return shells, functions


def make_basis_function(center: np.ndarray, ang: tuple[int, int, int], shell: ShellSpec) -> BasisFunction:
    exponents = np.asarray(shell.exponents, dtype=float)
    raw_coeffs = np.asarray(shell.coeffs, dtype=float)
    primitive_norms = np.asarray([primitive_normalization(alpha, ang) for alpha in exponents], dtype=float)
    coeffs = raw_coeffs * primitive_norms

    # Fallback to local scalar overlaps internally for static normalizations
    norm = 0.0
    for i, alpha in enumerate(exponents):
        for j, beta in enumerate(exponents):
            p = alpha + beta
            ex = hermite_coefficient(ang[0], ang[0], 0, 0.0, alpha, beta)
            ey = hermite_coefficient(ang[1], ang[1], 0, 0.0, alpha, beta)
            ez = hermite_coefficient(ang[2], ang[2], 0, 0.0, alpha, beta)
            norm += coeffs[i] * coeffs[j] * ex * ey * ez * (math.pi / p) ** 1.5
    coeffs /= math.sqrt(norm)
    return BasisFunction(center=center, ang=ang, exponents=exponents, coeffs=coeffs)


def primitive_normalization(alpha: float, ang: tuple[int, int, int]) -> float:
    l, m, n = ang
    denom = double_factorial(2 * l - 1) * double_factorial(2 * m - 1) * double_factorial(2 * n - 1)
    return (2.0 * alpha / math.pi) ** 0.75 * math.sqrt((4.0 * alpha) ** (l + m + n) / denom)


def double_factorial(value: int) -> int:
    if value <= 0: return 1
    result = 1
    for item in range(value, 0, -2): result *= item
    return result


@lru_cache(maxsize=None)
def hermite_coefficient(i: int, j: int, t: int, qx: float, alpha: float, beta: float) -> float:
    if t < 0 or t > i + j: return 0.0
    p = alpha + beta
    q = alpha * beta / p
    if i == 0 and j == 0: return math.exp(-q * qx ** 2) if t == 0 else 0.0
    if j == 0:
        return (hermite_coefficient(i - 1, j, t - 1, qx, alpha, beta) / (2.0 * p)
                - q * qx * hermite_coefficient(i - 1, j, t, qx, alpha, beta) / alpha
                + (t + 1) * hermite_coefficient(i - 1, j, t + 1, qx, alpha, beta))
    return (hermite_coefficient(i, j - 1, t - 1, qx, alpha, beta) / (2.0 * p)
            + q * qx * hermite_coefficient(i, j - 1, t, qx, alpha, beta) / beta
            + (t + 1) * hermite_coefficient(i, j - 1, t + 1, qx, alpha, beta))


@torch.no_grad()
def build_integrals(
    atoms: list[Atom], basis_functions: list[BasisFunction], shells: list[Shell],
    device: torch.device = None, dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if device is None: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nbf = len(basis_functions)
    overlap_gpu = torch.zeros((nbf, nbf), dtype=dtype, device=device)
    core_h_gpu = torch.zeros((nbf, nbf), dtype=dtype, device=device)
    eri_gpu = torch.zeros((nbf, nbf, nbf, nbf), dtype=dtype, device=device)

    # ---------------- 1-Electron Integrals (Grouped & Padded limits) ----------------
    pair_groups = defaultdict(list)
    for i, shell_a in enumerate(shells):
        for j, shell_b in enumerate(shells[:i+1]):
            K_a, K_b = len(basis_functions[shell_a.basis_indices[0]].exponents), len(basis_functions[shell_b.basis_indices[0]].exponents)
            pair_groups[(shell_a.ang_momentum, shell_b.ang_momentum, K_a, K_b)].append((i, j))

    for (L_a, L_b, K_a, K_b), pairs in pair_groups.items():
        Q = len(pairs)
        K_total = K_a * K_b
        B_size = Q * K_total

        cart_A, cart_B = cartesian_tuples(L_a), cartesian_tuples(L_b)

        A_centers = torch.tensor(np.array([shells[q[0]].center for q in pairs]), device=device, dtype=dtype)
        B_centers = torch.tensor(np.array([shells[q[1]].center for q in pairs]), device=device, dtype=dtype)
        A_exp = torch.tensor(np.array([basis_functions[shells[q[0]].basis_indices[0]].exponents for q in pairs]), device=device, dtype=dtype)
        B_exp = torch.tensor(np.array([basis_functions[shells[q[1]].basis_indices[0]].exponents for q in pairs]), device=device, dtype=dtype)
        A_coef = torch.tensor(np.array([basis_functions[shells[q[0]].basis_indices[0]].coeffs for q in pairs]), device=device, dtype=dtype)
        B_coef = torch.tensor(np.array([basis_functions[shells[q[1]].basis_indices[0]].coeffs for q in pairs]), device=device, dtype=dtype)

        g_a, g_b = torch.meshgrid(torch.arange(K_a), torch.arange(K_b), indexing='ij')
        g_a, g_b = g_a.flatten(), g_b.flatten()

        alpha, beta = A_exp[:, g_a].reshape(-1), B_exp[:, g_b].reshape(-1)
        coeffs = (A_coef[:, g_a] * B_coef[:, g_b]).reshape(-1)

        A_c = A_centers.unsqueeze(1).expand(Q, K_total, 3).reshape(-1, 3)
        B_c = B_centers.unsqueeze(1).expand(Q, K_total, 3).reshape(-1, 3)

        p = alpha + beta
        P = (alpha[:, None] * A_c + beta[:, None] * B_c) / p[:, None]
        AB = A_c - B_c

        # Overlap / Kinetic Eval
        Ex = batched_hermite_tensor(L_a, L_b, AB[:, 0], alpha, beta, p)
        Ey = batched_hermite_tensor(L_a, L_b, AB[:, 1], alpha, beta, p)
        Ez = batched_hermite_tensor(L_a, L_b, AB[:, 2], alpha, beta, p)
        Ex_kin = batched_hermite_tensor(L_a, L_b + 2, AB[:, 0], alpha, beta, p)
        Ey_kin = batched_hermite_tensor(L_a, L_b + 2, AB[:, 1], alpha, beta, p)
        Ez_kin = batched_hermite_tensor(L_a, L_b + 2, AB[:, 2], alpha, beta, p)

        def get_overlap_1d(E, i, j): return E[:, i, j, 0] * torch.sqrt(math.pi / p)
        def get_kinetic_1d(E, i, j):
            t1 = beta * (2*j + 1) * get_overlap_1d(E, i, j)
            t2 = -2.0 * beta * beta * get_overlap_1d(E, i, j + 2)
            t3 = torch.zeros_like(t1)
            if j >= 2: t3 = -0.5 * j * (j - 1) * get_overlap_1d(E, i, j - 2)
            return t1 + t2 + t3

        S_abcd, T_abcd = torch.zeros((B_size, len(cart_A), len(cart_B)), dtype=dtype, device=device), torch.zeros((B_size, len(cart_A), len(cart_B)), dtype=dtype, device=device)
        for a, (lx_a, ly_a, lz_a) in enumerate(cart_A):
            for b, (lx_b, ly_b, lz_b) in enumerate(cart_B):
                ex, ey, ez = Ex[:, lx_a, lx_b, 0], Ey[:, ly_a, ly_b, 0], Ez[:, lz_a, lz_b, 0]
                S_abcd[:, a, b] = ex * ey * ez * (math.pi / p)**1.5 * coeffs

                sx, sy, sz = get_overlap_1d(Ex_kin, lx_a, lx_b), get_overlap_1d(Ey_kin, ly_a, ly_b), get_overlap_1d(Ez_kin, lz_a, lz_b)
                tx, ty, tz = get_kinetic_1d(Ex_kin, lx_a, lx_b), get_kinetic_1d(Ey_kin, ly_a, ly_b), get_kinetic_1d(Ez_kin, lz_a, lz_b)
                T_abcd[:, a, b] = (tx * sy * sz + sx * ty * sz + sx * sy * tz) * coeffs

        S_Q = S_abcd.view(Q, K_total, len(cart_A), len(cart_B)).sum(dim=1)
        T_Q = T_abcd.view(Q, K_total, len(cart_A), len(cart_B)).sum(dim=1)

        # Nuclear Repulsion Eval
        V_Q = torch.zeros((Q, len(cart_A), len(cart_B)), dtype=dtype, device=device)
        for atom in atoms:
            PC = P - torch.tensor(atom.coord, dtype=dtype, device=device)
            max_n = L_a + L_b
            R_nuc = compute_r_integrals_batched(max_n, max_n, max_n, p, PC, torch.sum(PC**2, dim=1))

            V_abcd = torch.zeros((B_size, len(cart_A), len(cart_B)), dtype=dtype, device=device)
            prefactor_nuc = -atom.charge * 2.0 * math.pi / p

            for a, (lx_a, ly_a, lz_a) in enumerate(cart_A):
                for b, (lx_b, ly_b, lz_b) in enumerate(cart_B):
                    R_slice = R_nuc[:, :lx_a+lx_b+1, :ly_a+ly_b+1, :lz_a+lz_b+1]
                    val = torch.einsum('bt, bu, bv, btuv -> b', Ex[:, lx_a, lx_b, :lx_a+lx_b+1], Ey[:, ly_a, ly_b, :ly_a+ly_b+1], Ez[:, lz_a, lz_b, :lz_a+lz_b+1], R_slice)
                    V_abcd[:, a, b] = val * prefactor_nuc * coeffs
            V_Q += V_abcd.view(Q, K_total, len(cart_A), len(cart_B)).sum(dim=1)

        a_idx = torch.tensor(np.array([shells[q[0]].basis_indices for q in pairs]), device=device)
        b_idx = torch.tensor(np.array([shells[q[1]].basis_indices for q in pairs]), device=device)
        A_flat = a_idx.unsqueeze(2).expand(Q, len(cart_A), len(cart_B)).reshape(-1)
        B_flat = b_idx.unsqueeze(1).expand(Q, len(cart_A), len(cart_B)).reshape(-1)
        S_flat, H_flat = S_Q.reshape(-1), (T_Q + V_Q).reshape(-1)

        overlap_gpu.index_put_((A_flat, B_flat), S_flat); overlap_gpu.index_put_((B_flat, A_flat), S_flat)
        core_h_gpu.index_put_((A_flat, B_flat), H_flat); core_h_gpu.index_put_((B_flat, A_flat), H_flat)

    eri_gpu = build_eri_tensor(
        basis_functions=basis_functions,
        shells=shells,
        cartesian_tuples=cartesian_tuples,
        device=device,
        dtype=dtype,
    )

    return overlap_gpu, core_h_gpu, eri_gpu


def nuclear_repulsion_energy(atoms: list[Atom]) -> float:
    energy = 0.0
    for i, atom_i in enumerate(atoms):
        for atom_j in atoms[:i]:
            distance = float(np.linalg.norm(atom_i.coord - atom_j.coord))
            energy += atom_i.charge * atom_j.charge / distance
    return energy


def symmetric_orthogonalizer(overlap: torch.Tensor) -> torch.Tensor:
    eigvals, eigvecs = torch.linalg.eigh(overlap)
    if float(torch.min(eigvals).item()) < 1.0e-10:
        raise ValueError("Overlap matrix is singular or ill-conditioned")
    inv_sqrt = torch.diag(eigvals ** -0.5)
    return eigvecs @ inv_sqrt @ eigvecs.T


def fibonacci_sphere(num_points: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(num_points, dtype=float)
    z = 1.0 - 2.0 * (idx + 0.5) / num_points
    phi = math.pi * (3.0 - math.sqrt(5.0)) * idx
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    points = np.stack([radius * np.cos(phi), radius * np.sin(phi), z], axis=1)
    weights = np.full(num_points, 4.0 * math.pi / num_points, dtype=float)
    return points, weights


def radial_grid(num_points: int, scale: float) -> tuple[np.ndarray, np.ndarray]:
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


@torch.no_grad()
def evaluate_basis_on_grid(
    basis_functions: list[BasisFunction],
    grid_points: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    points = torch.tensor(grid_points, device=device, dtype=dtype)
    ngrid = points.shape[0]
    nbf = len(basis_functions)
    values = torch.empty((ngrid, nbf), device=device, dtype=dtype)
    gradients = torch.empty((3, ngrid, nbf), device=device, dtype=dtype)

    for ibf, bf in enumerate(basis_functions):
        center = torch.tensor(bf.center, device=device, dtype=dtype)
        exponents = torch.tensor(bf.exponents, device=device, dtype=dtype).view(-1, 1)
        coeffs = torch.tensor(bf.coeffs, device=device, dtype=dtype).view(-1, 1)

        diff = points - center[None, :]
        dx, dy, dz = diff[:, 0], diff[:, 1], diff[:, 2]
        r2 = torch.sum(diff * diff, dim=1)

        lx, ly, lz = bf.ang
        x_l = (dx ** lx) if lx > 0 else torch.ones_like(dx)
        y_m = (dy ** ly) if ly > 0 else torch.ones_like(dy)
        z_n = (dz ** lz) if lz > 0 else torch.ones_like(dz)
        x_lm1 = (dx ** (lx - 1)) if lx > 0 else torch.ones_like(dx)
        y_mm1 = (dy ** (ly - 1)) if ly > 0 else torch.ones_like(dy)
        z_nm1 = (dz ** (lz - 1)) if lz > 0 else torch.ones_like(dz)

        exp_term = torch.exp(-exponents * r2.unsqueeze(0))
        poly = (x_l * y_m * z_n).unsqueeze(0)
        values[:, ibf] = torch.sum(coeffs * exp_term * poly, dim=0)

        dpoly_x = (lx * x_lm1 * y_m * z_n).unsqueeze(0) if lx > 0 else 0.0
        dpoly_y = (ly * x_l * y_mm1 * z_n).unsqueeze(0) if ly > 0 else 0.0
        dpoly_z = (lz * x_l * y_m * z_nm1).unsqueeze(0) if lz > 0 else 0.0

        gradients[0, :, ibf] = torch.sum(coeffs * exp_term * (dpoly_x - 2.0 * exponents * dx.unsqueeze(0) * poly), dim=0)
        gradients[1, :, ibf] = torch.sum(coeffs * exp_term * (dpoly_y - 2.0 * exponents * dy.unsqueeze(0) * poly), dim=0)
        gradients[2, :, ibf] = torch.sum(coeffs * exp_term * (dpoly_z - 2.0 * exponents * dz.unsqueeze(0) * poly), dim=0)

    return values, gradients


def density_and_gradient(
    density: torch.Tensor,
    values: torch.Tensor,
    gradients: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    tmp = values @ density
    rho = torch.sum(tmp * values, dim=1)

    grad_rho = torch.empty((3, values.shape[0]), device=density.device, dtype=density.dtype)
    for axis in range(3):
        grad_rho[axis] = 2.0 * torch.sum(tmp * gradients[axis], dim=1)
    return rho, grad_rho


def evaluate_xc_terms(
    density: torch.Tensor,
    values: torch.Tensor,
    gradients: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rho, grad_rho = density_and_gradient(density, values, gradients)
    grad_norm = torch.sqrt(torch.sum(grad_rho * grad_rho, dim=0))

    # Securely port logic from PyTorch back to the static B3LYP Backend (libxc limits via CPU NumPy)
    energy_density_np, vrho_np, vgrad_np = evaluate_b3lyp_xc(rho.cpu().numpy(), grad_norm.cpu().numpy())

    device, dtype = density.device, density.dtype
    energy_density = torch.tensor(energy_density_np, device=device, dtype=dtype)
    vrho = torch.tensor(vrho_np, device=device, dtype=dtype)
    vgrad = torch.tensor(vgrad_np, device=device, dtype=dtype)

    return grad_rho, grad_norm, energy_density, vrho, vgrad


def build_xc_matrix_and_energy(
    density: torch.Tensor,
    values: torch.Tensor,
    gradients: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    grad_rho, grad_norm, energy_density, vrho, vgrad = evaluate_xc_terms(density, values, gradients)

    vmat = values.T @ (weights[:, None] * vrho[:, None] * values)
    safe_grad_norm = torch.where(grad_norm > 1.0e-18, grad_norm, torch.ones_like(grad_norm))
    direction = grad_rho / safe_grad_norm[None, :]
    for axis in range(3):
        factor = weights * vgrad * direction[axis]
        mixed = gradients[axis].T @ (factor[:, None] * values)
        vmat += mixed + mixed.T

    return vmat, float(torch.sum(weights * energy_density).item())


def build_xc_energy(
    density: torch.Tensor,
    values: torch.Tensor,
    gradients: torch.Tensor,
    weights: torch.Tensor,
) -> float:
    _, _, energy_density, _, _ = evaluate_xc_terms(density, values, gradients)
    return float(torch.sum(weights * energy_density).item())


def initial_density(
    core_h: torch.Tensor,
    orthogonalizer: torch.Tensor,
    nocc: int,
) -> torch.Tensor:
    fock_ortho = orthogonalizer.T @ core_h @ orthogonalizer
    _, coeffs_ortho = torch.linalg.eigh(fock_ortho)
    coeffs = orthogonalizer @ coeffs_ortho
    coeffs_occ = coeffs[:, :nocc]
    return 2.0 * (coeffs_occ @ coeffs_occ.T)


@torch.no_grad()
def restricted_scf(
    method: str,
    core_h: torch.Tensor,
    overlap: torch.Tensor,
    eri: torch.Tensor,
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
    device, dtype = core_h.device, core_h.dtype
    nocc = nelec // 2
    exact_exchange_fraction = 1.0 if method == "rhf" else B3LYP_EXACT_EXCHANGE

    x_gpu = symmetric_orthogonalizer(overlap)
    density = initial_density(core_h, x_gpu, nocc)
    diis = DIIS()

    values = gradients = weights_gpu = None
    if method == "b3lyp":
        if grid_points is None or grid_weights is None:
            raise ValueError("B3LYP requires a numerical integration grid")
        values, gradients = evaluate_basis_on_grid(basis_functions, grid_points, device, dtype)
        weights_gpu = torch.tensor(grid_weights, device=device, dtype=dtype)

    energy_prev: float | None = None
    history: list[dict[str, float]] = []
    converged = False
    orbital_energies = torch.tensor([], dtype=dtype)

    for cycle in range(1, max_cycle + 1):
        j_mat = torch.einsum("pqrs,rs->pq", eri, density)
        k_mat = torch.einsum("prqs,rs->pq", eri, density)
        xc_matrix = torch.zeros_like(core_h)
        e_xc = 0.0

        if method == "b3lyp":
            xc_matrix, e_xc = build_xc_matrix_and_energy(density, values, gradients, weights_gpu)

        fock = core_h + j_mat - 0.5 * exact_exchange_fraction * k_mat + xc_matrix
        fock = diis.extrapolate(fock, density, overlap)

        fock_ortho = x_gpu.T @ fock @ x_gpu
        orbital_energies, coeffs_ortho = torch.linalg.eigh(fock_ortho)
        coeffs = x_gpu @ coeffs_ortho
        coeffs_occ = coeffs[:, :nocc]
        density_new = 2.0 * (coeffs_occ @ coeffs_occ.T)

        j_new = torch.einsum("pqrs,rs->pq", eri, density_new)
        k_new = torch.einsum("prqs,rs->pq", eri, density_new)

        e_xc_new = 0.0
        if method == "b3lyp":
            e_xc_new = build_xc_energy(density_new, values, gradients, weights_gpu)

        e_one = float(torch.sum(density_new * core_h).item())
        e_coulomb = 0.5 * float(torch.sum(density_new * j_new).item())
        e_exact_exchange = -0.25 * exact_exchange_fraction * float(torch.sum(density_new * k_new).item())
        total_energy = e_one + e_coulomb + e_exact_exchange + e_xc_new + e_nuc

        delta_e = math.inf if energy_prev is None else abs(total_energy - energy_prev)
        rms_density = float(torch.sqrt(torch.mean((density_new - density) ** 2)).item())

        history.append({
            "cycle": float(cycle),
            "energy_hartree": total_energy,
            "delta_e": delta_e,
            "rms_density": rms_density,
        })

        if verbose:
            print(f"cycle={cycle:02d} E={total_energy:.12f} dE={delta_e:.3e} rmsD={rms_density:.3e}")

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
        "orbital_energies_hartree": orbital_energies.cpu().numpy().tolist(),
    }


def main() -> None:
    args = parse_args()
    parsed_basis = parse_orca_basis(args.basis)
    geometry = get_geometry(args)
    atoms, nelec = build_atoms(geometry, args.charge)
    shells, basis_functions = build_shells_and_basis_functions(atoms, parsed_basis)

    if args.verbose:
        print(f"building one- and two-electron integrals for {len(basis_functions)} basis functions")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    overlap, core_h, eri = build_integrals(atoms, basis_functions, shells, device, dtype)
    e_nuc = nuclear_repulsion_energy(atoms)

    grid_points = grid_weights = None
    if args.method == "b3lyp":
        if args.verbose:
            print(f"using B3LYP XC backend: {get_b3lyp_backend()}")
            print(f"building multicenter grid with {args.grid_radial} radial x {args.grid_angular} angular points per atom")
        grid_points, grid_weights = build_grid(
            atoms=atoms, num_radial=args.grid_radial, num_angular=args.grid_angular, partition_power=args.partition_power,
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
        "system": [{"symbol": atom.symbol, "coord_bohr": atom.coord.tolist()} for atom in atoms],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
