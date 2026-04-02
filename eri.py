from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable, Sequence

import numpy as np
import torch

import triton
import triton.language as tl


@triton.jit
def mako_gemm_coalescing_kernel(
    e_ab_ptr, pq_ptr, e_cd_ptr, abcd_ptr,
    M, K1, K2, N,
    stride_ab_m, stride_ab_k1,
    stride_pq_k1, stride_pq_k2,
    stride_cd_k2, stride_cd_n,
    stride_out_m, stride_out_n,
    BLOCK_M: tl.constexpr,
    BLOCK_K1: tl.constexpr,
    BLOCK_K2: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k1 = tl.arange(0, BLOCK_K1)
    offs_k2 = tl.arange(0, BLOCK_K2)

    e_ab_ptrs = e_ab_ptr + (offs_m[:, None] * stride_ab_m + offs_k1[None, :] * stride_ab_k1)
    pq_ptrs = pq_ptr + (offs_k1[:, None] * stride_pq_k1 + offs_k2[None, :] * stride_pq_k2)

    e_ab = tl.load(e_ab_ptrs).to(tl.float16)
    pq = tl.load(pq_ptrs).to(tl.float16)
    ab_q = tl.dot(e_ab, pq, out_dtype=tl.float32)
    ab_q_fp16 = ab_q.to(tl.float16)

    e_cd_ptrs = e_cd_ptr + (offs_k2[:, None] * stride_cd_k2 + offs_n[None, :] * stride_cd_n)
    e_cd = tl.load(e_cd_ptrs).to(tl.float16)
    ab_cd = tl.dot(ab_q_fp16, e_cd, out_dtype=tl.float32)

    out_ptrs = abcd_ptr + (offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n)
    tl.store(out_ptrs, ab_cd.to(tl.float64))


def boys_gpu(n: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    mask = x > 1e-8
    x_safe = torch.where(mask, x, torch.ones_like(x) * 1e-8)
    a = n + 0.5
    gamma_a = torch.exp(torch.lgamma(a))
    val = gamma_a * torch.special.gammainc(a, x_safe) / (2.0 * x_safe ** a)
    taylor = 1.0 / (2.0 * n + 1.0) - x / (2.0 * n + 3.0)
    return torch.where(mask, val, taylor)


def compute_r_integrals_batched(
    max_t: int,
    max_u: int,
    max_v: int,
    alpha_eri: torch.Tensor,
    pq: torch.Tensor,
    rpq2: torch.Tensor,
) -> torch.Tensor:
    batch_size = alpha_eri.shape[0]
    device, dtype = alpha_eri.device, alpha_eri.dtype

    def build_c(max_val: int, pc: torch.Tensor) -> torch.Tensor:
        c_tensor = torch.zeros((batch_size, max_val + 1, max_val + 1), dtype=dtype, device=device)
        c_tensor[:, 0, 0] = 1.0
        if max_val > 0:
            c_tensor[:, 1, 1] = pc
        for k in range(2, max_val + 1):
            c_tensor[:, k, 1:] = pc.unsqueeze(1) * c_tensor[:, k - 1, :-1] + (k - 1) * c_tensor[:, k - 2, :-1]
        return c_tensor

    cx = build_c(max_t, pq[:, 0])
    cy = build_c(max_u, pq[:, 1])
    cz = build_c(max_v, pq[:, 2])

    max_n = max_t + max_u + max_v
    n_idx = torch.arange(max_n + 1, device=device, dtype=dtype)
    n_exp = n_idx.unsqueeze(1)
    p_exp = alpha_eri.unsqueeze(0)
    x_exp = (alpha_eri * rpq2).unsqueeze(0)

    boys_vals = boys_gpu(n_exp, x_exp)
    sign_n = torch.where(
        n_idx % 2 == 0,
        torch.tensor(1.0, dtype=dtype, device=device),
        torch.tensor(-1.0, dtype=dtype, device=device),
    )
    f_tensor = (sign_n.unsqueeze(1) * ((2.0 * p_exp) ** n_exp) * boys_vals).t()

    i = torch.arange(max_t + 1, device=device)[:, None, None]
    j = torch.arange(max_u + 1, device=device)[None, :, None]
    k = torch.arange(max_v + 1, device=device)[None, None, :]
    return torch.einsum("bti,buj,bvk,bijk->btuv", cx, cy, cz, f_tensor[:, i + j + k])


def batched_hermite_tensor(
    max_ang_a: int,
    max_ang_b: int,
    a_minus_b: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    p: torch.Tensor,
) -> torch.Tensor:
    batch_size = alpha.shape[0]
    device, dtype = alpha.device, alpha.dtype
    hermite = torch.zeros(
        (max_ang_a + 1, max_ang_b + 1, max_ang_a + max_ang_b + 1, batch_size),
        dtype=dtype,
        device=device,
    )
    hermite[0, 0, 0, :] = torch.exp(-((alpha * beta) / p) * (a_minus_b ** 2))

    for i in range(max_ang_a + 1):
        for j in range(max_ang_b + 1):
            if i == 0 and j == 0:
                continue
            for t in range(i + j + 1):
                value = torch.zeros(batch_size, dtype=dtype, device=device)
                if i > 0:
                    if t > 0:
                        value += (1.0 / (2.0 * p)) * hermite[i - 1, j, t - 1, :]
                    value += -(beta / p) * a_minus_b * hermite[i - 1, j, t, :]
                    if t + 1 <= i + j - 1:
                        value += (t + 1) * hermite[i - 1, j, t + 1, :]
                else:
                    if t > 0:
                        value += (1.0 / (2.0 * p)) * hermite[i, j - 1, t - 1, :]
                    value += (alpha / p) * a_minus_b * hermite[i, j - 1, t, :]
                    if t + 1 <= i + j - 1:
                        value += (t + 1) * hermite[i, j - 1, t + 1, :]
                hermite[i, j, t, :] = value
    return hermite.permute(3, 0, 1, 2)


def _shell_primitive_arrays(
    quartets: list[tuple[int, int, int, int]],
    basis_functions: Sequence[Any],
    shells: Sequence[Any],
    center_index: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shell_indices = [entry[center_index] for entry in quartets]
    shell_basis_indices = [shells[idx].basis_indices[0] for idx in shell_indices]
    centers = torch.tensor(np.array([shells[idx].center for idx in shell_indices]), device=device, dtype=dtype)
    exponents = torch.tensor(
        np.array([basis_functions[idx].exponents for idx in shell_basis_indices]),
        device=device,
        dtype=dtype,
    )
    coeffs = torch.tensor(
        np.array([basis_functions[idx].coeffs for idx in shell_basis_indices]),
        device=device,
        dtype=dtype,
    )
    return centers, exponents, coeffs


@torch.no_grad()
def build_eri_tensor(
    basis_functions: Sequence[Any],
    shells: Sequence[Any],
    cartesian_tuples: Callable[[int], tuple[tuple[int, int, int], ...]],
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    nbf = len(basis_functions)
    eri_gpu = torch.zeros((nbf, nbf, nbf, nbf), dtype=dtype, device=device)

    quartet_groups: dict[tuple[int, ...], list[tuple[int, int, int, int]]] = defaultdict(list)
    for i, shell_a in enumerate(shells):
        for j, shell_b in enumerate(shells[: i + 1]):
            ij = i * (i + 1) // 2 + j
            for k, shell_c in enumerate(shells):
                for l, shell_d in enumerate(shells[: k + 1]):
                    if ij < k * (k + 1) // 2 + l:
                        continue
                    k_a = len(basis_functions[shell_a.basis_indices[0]].exponents)
                    k_b = len(basis_functions[shell_b.basis_indices[0]].exponents)
                    k_c = len(basis_functions[shell_c.basis_indices[0]].exponents)
                    k_d = len(basis_functions[shell_d.basis_indices[0]].exponents)
                    quartet_groups[
                        (
                            shell_a.ang_momentum,
                            shell_b.ang_momentum,
                            shell_c.ang_momentum,
                            shell_d.ang_momentum,
                            k_a,
                            k_b,
                            k_c,
                            k_d,
                        )
                    ].append((i, j, k, l))

    for key, quartets in quartet_groups.items():
        quartet_count = len(quartets)
        l_a, l_b, l_c, l_d, k_a, k_b, k_c, k_d = key
        primitive_count = k_a * k_b * k_c * k_d
        cart_a = cartesian_tuples(l_a)
        cart_b = cartesian_tuples(l_b)
        cart_c = cartesian_tuples(l_c)
        cart_d = cartesian_tuples(l_d)

        def get_idx(cart: tuple[tuple[int, int, int], ...], dim: int) -> torch.Tensor:
            return torch.tensor(np.array([ang[dim] for ang in cart]), dtype=torch.long, device=device)

        idx_a = [get_idx(cart_a, dim) for dim in range(3)]
        idx_b = [get_idx(cart_b, dim) for dim in range(3)]
        idx_c_x, idx_d_x = torch.meshgrid(get_idx(cart_c, 0), get_idx(cart_d, 0), indexing="ij")
        idx_c_y, idx_d_y = torch.meshgrid(get_idx(cart_c, 1), get_idx(cart_d, 1), indexing="ij")
        idx_c_z, idx_d_z = torch.meshgrid(get_idx(cart_c, 2), get_idx(cart_d, 2), indexing="ij")

        a_centers, a_exp, a_coef = _shell_primitive_arrays(quartets, basis_functions, shells, 0, device, dtype)
        b_centers, b_exp, b_coef = _shell_primitive_arrays(quartets, basis_functions, shells, 1, device, dtype)
        c_centers, c_exp, c_coef = _shell_primitive_arrays(quartets, basis_functions, shells, 2, device, dtype)
        d_centers, d_exp, d_coef = _shell_primitive_arrays(quartets, basis_functions, shells, 3, device, dtype)

        g_a, g_b, g_c, g_d = torch.meshgrid(
            torch.arange(k_a),
            torch.arange(k_b),
            torch.arange(k_c),
            torch.arange(k_d),
            indexing="ij",
        )
        g_a = g_a.flatten()
        g_b = g_b.flatten()
        g_c = g_c.flatten()
        g_d = g_d.flatten()

        max_t_all = l_a + l_b + l_c + l_d + 1
        bytes_per_prim = 8 * (
            3 * (l_a + l_b + 1) ** 2
            + 3 * (l_c + l_d + 1) ** 2
            + 3 * (l_a + 1) * (l_b + 1) * (l_c + 1) * (l_d + 1) * max_t_all
            + max_t_all ** 3
            + len(cart_a) * len(cart_b) * len(cart_c) * len(cart_d)
        )
        quartet_chunk_size = max(1, max(1, (512 * 1024 * 1024) // bytes_per_prim) // primitive_count)

        quartets_eri = torch.zeros(
            (quartet_count, len(cart_a), len(cart_b), len(cart_c), len(cart_d)),
            dtype=dtype,
            device=device,
        )

        for chunk_start in range(0, quartet_count, quartet_chunk_size):
            chunk_end = min(quartet_count, chunk_start + quartet_chunk_size)
            chunk_quartets = chunk_end - chunk_start
            batch_size = chunk_quartets * primitive_count

            alpha = a_exp[chunk_start:chunk_end][:, g_a].reshape(-1)
            beta = b_exp[chunk_start:chunk_end][:, g_b].reshape(-1)
            gamma = c_exp[chunk_start:chunk_end][:, g_c].reshape(-1)
            delta = d_exp[chunk_start:chunk_end][:, g_d].reshape(-1)
            coeffs = (
                a_coef[chunk_start:chunk_end][:, g_a]
                * b_coef[chunk_start:chunk_end][:, g_b]
                * c_coef[chunk_start:chunk_end][:, g_c]
                * d_coef[chunk_start:chunk_end][:, g_d]
            ).reshape(-1)

            a_center = a_centers[chunk_start:chunk_end].unsqueeze(1).expand(chunk_quartets, primitive_count, 3).reshape(-1, 3)
            b_center = b_centers[chunk_start:chunk_end].unsqueeze(1).expand(chunk_quartets, primitive_count, 3).reshape(-1, 3)
            c_center = c_centers[chunk_start:chunk_end].unsqueeze(1).expand(chunk_quartets, primitive_count, 3).reshape(-1, 3)
            d_center = d_centers[chunk_start:chunk_end].unsqueeze(1).expand(chunk_quartets, primitive_count, 3).reshape(-1, 3)

            p = alpha + beta
            q = gamma + delta
            p_center = (alpha[:, None] * a_center + beta[:, None] * b_center) / p[:, None]
            q_center = (gamma[:, None] * c_center + delta[:, None] * d_center) / q[:, None]
            pq = p_center - q_center

            ex_ab = batched_hermite_tensor(l_a, l_b, a_center[:, 0] - b_center[:, 0], alpha, beta, p)
            ey_ab = batched_hermite_tensor(l_a, l_b, a_center[:, 1] - b_center[:, 1], alpha, beta, p)
            ez_ab = batched_hermite_tensor(l_a, l_b, a_center[:, 2] - b_center[:, 2], alpha, beta, p)
            ex_cd = batched_hermite_tensor(l_c, l_d, c_center[:, 0] - d_center[:, 0], gamma, delta, q)
            ey_cd = batched_hermite_tensor(l_c, l_d, c_center[:, 1] - d_center[:, 1], gamma, delta, q)
            ez_cd = batched_hermite_tensor(l_c, l_d, c_center[:, 2] - d_center[:, 2], gamma, delta, q)

            def convolve(e_ab: torch.Tensor, e_cd: torch.Tensor) -> torch.Tensor:
                _, i_dim, j_dim, t_dim = e_ab.shape
                _, k_dim, l_dim, tau_dim = e_cd.shape
                g_tensor = torch.zeros(
                    (batch_size, i_dim, j_dim, k_dim, l_dim, t_dim + tau_dim - 1),
                    dtype=dtype,
                    device=device,
                )
                for t in range(t_dim):
                    for tau in range(tau_dim):
                        sign = 1.0 if tau % 2 == 0 else -1.0
                        g_tensor[..., t + tau] += (
                            e_ab[..., t].unsqueeze(3).unsqueeze(4)
                            * e_cd[..., tau].unsqueeze(1).unsqueeze(2)
                            * sign
                        )
                return g_tensor

            gx = convolve(ex_ab, ex_cd)
            gy = convolve(ey_ab, ey_cd)
            gz = convolve(ez_ab, ez_cd)
            r_tensor = compute_r_integrals_batched(
                max_t_all - 1,
                max_t_all - 1,
                max_t_all - 1,
                p * q / (p + q),
                pq,
                torch.sum(pq ** 2, dim=1),
            )
            weighted_coeffs = coeffs * (2.0 * math.pi ** 2.5 / (p * q * torch.sqrt(p + q)))

            batch_eri = torch.zeros(
                (batch_size, len(cart_a), len(cart_b), len(cart_c), len(cart_d)),
                dtype=dtype,
                device=device,
            )
            for a in range(len(cart_a)):
                for b in range(len(cart_b)):
                    gx_slice = gx[:, idx_a[0][a], idx_b[0][b], idx_c_x, idx_d_x, :]
                    gy_slice = gy[:, idx_a[1][a], idx_b[1][b], idx_c_y, idx_d_y, :]
                    gz_slice = gz[:, idx_a[2][a], idx_b[2][b], idx_c_z, idx_d_z, :]
                    batch_eri[:, a, b, :, :] = (
                        torch.einsum("ncdt,ncdu,ncdv,ntuv->ncd", gx_slice, gy_slice, gz_slice, r_tensor)
                        * weighted_coeffs[:, None, None]
                    )

            quartets_eri[chunk_start:chunk_end] = batch_eri.view(
                chunk_quartets,
                primitive_count,
                len(cart_a),
                len(cart_b),
                len(cart_c),
                len(cart_d),
            ).sum(dim=1)

        a_idx = torch.tensor(np.array([shells[entry[0]].basis_indices for entry in quartets]), device=device)
        b_idx = torch.tensor(np.array([shells[entry[1]].basis_indices for entry in quartets]), device=device)
        c_idx = torch.tensor(np.array([shells[entry[2]].basis_indices for entry in quartets]), device=device)
        d_idx = torch.tensor(np.array([shells[entry[3]].basis_indices for entry in quartets]), device=device)

        a_mesh = a_idx.view(quartet_count, len(cart_a), 1, 1, 1).expand(quartet_count, len(cart_a), len(cart_b), len(cart_c), len(cart_d))
        b_mesh = b_idx.view(quartet_count, 1, len(cart_b), 1, 1).expand(quartet_count, len(cart_a), len(cart_b), len(cart_c), len(cart_d))
        c_mesh = c_idx.view(quartet_count, 1, 1, len(cart_c), 1).expand(quartet_count, len(cart_a), len(cart_b), len(cart_c), len(cart_d))
        d_mesh = d_idx.view(quartet_count, 1, 1, 1, len(cart_d)).expand(quartet_count, len(cart_a), len(cart_b), len(cart_c), len(cart_d))

        values = quartets_eri.reshape(-1)
        a_flat = a_mesh.reshape(-1)
        b_flat = b_mesh.reshape(-1)
        c_flat = c_mesh.reshape(-1)
        d_flat = d_mesh.reshape(-1)

        eri_gpu.index_put_((a_flat, b_flat, c_flat, d_flat), values)
        eri_gpu.index_put_((b_flat, a_flat, c_flat, d_flat), values)
        eri_gpu.index_put_((a_flat, b_flat, d_flat, c_flat), values)
        eri_gpu.index_put_((b_flat, a_flat, d_flat, c_flat), values)
        eri_gpu.index_put_((c_flat, d_flat, a_flat, b_flat), values)
        eri_gpu.index_put_((d_flat, c_flat, a_flat, b_flat), values)
        eri_gpu.index_put_((c_flat, d_flat, b_flat, a_flat), values)
        eri_gpu.index_put_((d_flat, c_flat, b_flat, a_flat), values)

    return eri_gpu
