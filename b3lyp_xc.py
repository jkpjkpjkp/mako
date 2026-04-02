from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cupy as cp
import numpy as np

Array = np.ndarray | cp.ndarray


B3LYP_EXACT_EXCHANGE = 0.20
B88_EXCHANGE_WEIGHT = 0.72
SLATER_EXCHANGE_WEIGHT = 0.08

RHO_FLOOR = 1.0e-14
RHO_TABLE_MIN = 1.0e-12
RHO_TABLE_MAX = 1.0e3
REDUCED_GRADIENT_MAX = 200.0

CX = 0.75 * (3.0 / np.pi) ** (1.0 / 3.0)
B88_BETA = 0.0042
TWO_POW_ONE_THIRD = 2.0 ** (1.0 / 3.0)


@lru_cache(maxsize=1)
def _load_libxc():
    try:
        from pyscf.dft import libxc
    except Exception:
        return None
    return libxc


def get_b3lyp_backend() -> str:
    return "pyscf-libxc" if _load_libxc() is not None else "table"


@lru_cache(maxsize=1)
def _load_correlation_table() -> dict[str, cp.ndarray | float | int]:
    path = Path(__file__).with_name("b3lyp_correlation_table.npz")
    if not path.exists():
        raise FileNotFoundError(
            f"Missing baked B3LYP correlation table: {path}. "
            "Regenerate it if the repository checkout omitted binary artifacts."
        )

    with np.load(path) as data:
        corr_f = cp.asarray(data["corr_f"], dtype=cp.float64)
        corr_vrho = cp.asarray(data["corr_vrho"], dtype=cp.float64)
        corr_vgrad = cp.asarray(data["corr_vgrad"], dtype=cp.float64)
        nrho, nq = corr_f.shape

    return {
        "corr_f": corr_f,
        "corr_vrho": corr_vrho,
        "corr_vgrad": corr_vgrad,
        "nrho": int(nrho),
        "nq": int(nq),
        "log_rho_min": float(np.log(RHO_TABLE_MIN)),
        "log_rho_span": float(np.log(RHO_TABLE_MAX) - np.log(RHO_TABLE_MIN)),
        "log_q_span": float(np.log1p(REDUCED_GRADIENT_MAX)),
    }


def _exchange_energy_density(rho: cp.ndarray, grad_norm: cp.ndarray) -> cp.ndarray:
    rho_safe = cp.maximum(rho, RHO_FLOOR)
    rho_four_thirds = rho_safe ** (4.0 / 3.0)
    f_lda = -CX * rho_four_thirds

    x = TWO_POW_ONE_THIRD * grad_norm / rho_four_thirds
    denom = 1.0 + 6.0 * B88_BETA * x * cp.arcsinh(x)
    delta_b88 = -B88_BETA * TWO_POW_ONE_THIRD * grad_norm * grad_norm / (rho_four_thirds * denom)

    return (SLATER_EXCHANGE_WEIGHT + B88_EXCHANGE_WEIGHT) * f_lda + B88_EXCHANGE_WEIGHT * delta_b88


def _exchange_vrho(rho: cp.ndarray, grad_norm: cp.ndarray) -> cp.ndarray:
    rho_safe = cp.maximum(rho, RHO_FLOOR)
    step = cp.maximum(1.0e-8 * rho_safe, 1.0e-10)
    forward = _exchange_energy_density(rho_safe + step, grad_norm)
    backward = _exchange_energy_density(cp.maximum(rho_safe - step, RHO_FLOOR), grad_norm)
    return (forward - backward) / (2.0 * step)


def _exchange_vgrad(rho: cp.ndarray, grad_norm: cp.ndarray) -> cp.ndarray:
    rho_safe = cp.maximum(rho, RHO_FLOOR)
    rho_four_thirds = rho_safe ** (4.0 / 3.0)
    x_prefactor = TWO_POW_ONE_THIRD / rho_four_thirds
    x = x_prefactor * grad_norm
    denom = 1.0 + 6.0 * B88_BETA * x * cp.arcsinh(x)
    denom_prime = 6.0 * B88_BETA * x_prefactor * (
        cp.arcsinh(x) + x / cp.sqrt(1.0 + x * x)
    )
    beta_prefactor = B88_BETA * TWO_POW_ONE_THIRD / rho_four_thirds
    delta_prime = -beta_prefactor * (
        2.0 * grad_norm / denom
        - grad_norm * grad_norm * denom_prime / (denom * denom)
    )
    return B88_EXCHANGE_WEIGHT * delta_prime


def _interpolate_correlation_table(
    table: cp.ndarray,
    rho: cp.ndarray,
    grad_norm: cp.ndarray,
) -> cp.ndarray:
    table_spec = _load_correlation_table()
    rho_clipped = cp.clip(rho, RHO_TABLE_MIN, RHO_TABLE_MAX)
    reduced_gradient = cp.clip(
        grad_norm / (rho_clipped ** (4.0 / 3.0)),
        0.0,
        REDUCED_GRADIENT_MAX,
    )

    x_rho = (
        (cp.log(rho_clipped) - table_spec["log_rho_min"]) / table_spec["log_rho_span"]
    ) * (table_spec["nrho"] - 1)
    x_q = (cp.log1p(reduced_gradient) / table_spec["log_q_span"]) * (table_spec["nq"] - 1)

    i_rho = cp.clip(cp.floor(x_rho).astype(cp.int32), 0, table_spec["nrho"] - 2)
    i_q = cp.clip(cp.floor(x_q).astype(cp.int32), 0, table_spec["nq"] - 2)
    t_rho = x_rho - i_rho
    t_q = x_q - i_q

    a = table[i_rho, i_q]
    b = table[i_rho + 1, i_q]
    c = table[i_rho, i_q + 1]
    d = table[i_rho + 1, i_q + 1]
    return (
        (1.0 - t_rho) * (1.0 - t_q) * a
        + t_rho * (1.0 - t_q) * b
        + (1.0 - t_rho) * t_q * c
        + t_rho * t_q * d
    )


def _evaluate_b3lyp_xc_table(
    rho: cp.ndarray,
    grad_norm: cp.ndarray,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    rho_safe = cp.maximum(rho, 0.0)

    exchange_f = _exchange_energy_density(rho_safe, grad_norm)
    exchange_vrho = _exchange_vrho(rho_safe, grad_norm)
    exchange_vgrad = _exchange_vgrad(rho_safe, grad_norm)

    table_spec = _load_correlation_table()
    corr_f = _interpolate_correlation_table(table_spec["corr_f"], rho_safe, grad_norm)
    corr_vrho = _interpolate_correlation_table(table_spec["corr_vrho"], rho_safe, grad_norm)
    corr_vgrad = _interpolate_correlation_table(table_spec["corr_vgrad"], rho_safe, grad_norm)

    active = rho_safe > RHO_FLOOR
    energy_density = cp.where(active, exchange_f + corr_f, 0.0)
    vrho = cp.where(active, exchange_vrho + corr_vrho, 0.0)
    vgrad = cp.where(active, exchange_vgrad + corr_vgrad, 0.0)
    return energy_density, vrho, vgrad


def _evaluate_b3lyp_xc_libxc(
    rho: cp.ndarray,
    grad_norm: cp.ndarray,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    libxc = _load_libxc()
    if libxc is None:
        raise RuntimeError("PySCF libxc backend is unavailable")

    rho_safe = cp.maximum(rho, 0.0)
    grad_safe = cp.maximum(grad_norm, 0.0)
    rho_np = cp.asnumpy(rho_safe)
    grad_np = cp.asnumpy(grad_safe)
    zeros = np.zeros_like(rho_np)

    exc, vxc, _, _ = libxc.eval_xc(
        "B3LYP",
        (rho_np, grad_np, zeros, zeros),
        spin=0,
        deriv=1,
    )

    energy_density = rho_np * exc
    vrho = np.asarray(vxc[0], dtype=np.float64)
    # LibXC returns d(rho * exc) / d sigma with sigma = |grad rho|^2.
    vgrad = 2.0 * grad_np * np.asarray(vxc[1], dtype=np.float64)

    active = rho_np > RHO_FLOOR
    energy_density = np.where(active, energy_density, 0.0)
    vrho = np.where(active, vrho, 0.0)
    vgrad = np.where(active, vgrad, 0.0)
    return (
        cp.asarray(energy_density, dtype=cp.float64),
        cp.asarray(vrho, dtype=cp.float64),
        cp.asarray(vgrad, dtype=cp.float64),
    )


def _normalize_xc_inputs(rho: Array, grad_norm: Array) -> tuple[cp.ndarray, cp.ndarray, bool]:
    return_numpy = isinstance(rho, np.ndarray) or isinstance(grad_norm, np.ndarray)
    rho_cp = cp.asarray(rho, dtype=cp.float64)
    grad_norm_cp = cp.asarray(grad_norm, dtype=cp.float64)
    return rho_cp, grad_norm_cp, return_numpy


def _restore_xc_outputs(
    energy_density: cp.ndarray,
    vrho: cp.ndarray,
    vgrad: cp.ndarray,
    *,
    return_numpy: bool,
) -> tuple[Array, Array, Array]:
    if return_numpy:
        return cp.asnumpy(energy_density), cp.asnumpy(vrho), cp.asnumpy(vgrad)
    return energy_density, vrho, vgrad


def evaluate_b3lyp_xc(
    rho: Array,
    grad_norm: Array,
) -> tuple[Array, Array, Array]:
    rho_cp, grad_norm_cp, return_numpy = _normalize_xc_inputs(rho, grad_norm)
    if _load_libxc() is not None:
        energy_density, vrho, vgrad = _evaluate_b3lyp_xc_libxc(rho_cp, grad_norm_cp)
    else:
        energy_density, vrho, vgrad = _evaluate_b3lyp_xc_table(rho_cp, grad_norm_cp)
    return _restore_xc_outputs(
        energy_density,
        vrho,
        vgrad,
        return_numpy=return_numpy,
    )
