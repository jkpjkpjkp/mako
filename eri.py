import triton
import triton.language as tl

@triton.jit
def mako_gemm_coalescing_kernel(
    # Tensor Pointers
    e_ab_ptr, pq_ptr, e_cd_ptr, abcd_ptr,
    # Matrix dimensions (M=N_ab, K1=N_p, K2=N_q, N=N_cd)
    M, K1, K2, N,
    # Memory Strides
    stride_ab_m, stride_ab_k1,
    stride_pq_k1, stride_pq_k2,
    stride_cd_k2, stride_cd_n,
    stride_out_m, stride_out_n,
    # CompilerMako tunable block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_K1: tl.constexpr,
    BLOCK_K2: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    """
    Tile IR implementation of Algorithm 1 focusing on KernelMako's GEMM Coalescing.
    Fuses two basis transformations into a single hardware execution kernel.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k1 = tl.arange(0, BLOCK_K1)
    offs_k2 = tl.arange(0, BLOCK_K2)

    # -------------------------------------------------------------
    # MATMUL 1: (ab|q] = E_AB @ [p|q]
    # -------------------------------------------------------------
    e_ab_ptrs = e_ab_ptr + (offs_m[:, None] * stride_ab_m + offs_k1[None, :] * stride_ab_k1)
    pq_ptrs = pq_ptr + (offs_k1[:, None] * stride_pq_k1 + offs_k2[None, :] * stride_pq_k2)

    # QuantMako (Sec 3.2.1): Load inputs directly in FP16 to trigger Tensor Cores
    e_ab = tl.load(e_ab_ptrs).to(tl.float16)
    pq = tl.load(pq_ptrs).to(tl.float16)

    # QuantMako (Sec 3.2.2): Accumulate MatMul natively in FP32 registers
    ab_q = tl.dot(e_ab, pq, out_dtype=tl.float32)

    # -------------------------------------------------------------
    # KernelMako GEMM Coalescing (Sec 3.1.3)
    # -------------------------------------------------------------
    # The intermediate array `ab_q` is NEVER written to Global Memory (VRAM).
    # It stays strictly in Warp Registers! We simply cast it back to FP16
    # natively on the chip to feed the hardware for MatMul 2.
    ab_q_fp16 = ab_q.to(tl.float16)

    # -------------------------------------------------------------
    # MATMUL 2: (ab|cd) = (ab|q] @ E_CD
    # -------------------------------------------------------------
    e_cd_ptrs = e_cd_ptr + (offs_k2[:, None] * stride_cd_k2 + offs_n[None, :] * stride_cd_n)
    e_cd = tl.load(e_cd_ptrs).to(tl.float16)

    # Second MatMul, accumulate in high precision FP32
    ab_cd = tl.dot(ab_q_fp16, e_cd, out_dtype=tl.float32)

    # -------------------------------------------------------------
    # Final Stage: Write the computed ERI block
    # -------------------------------------------------------------
    # Cast to FP64 before writing to Global Memory to ensure exact
    # chemical accuracy when constructing the final Fock matrix.
    out_ptrs = abcd_ptr + (offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n)
    tl.store(out_ptrs, ab_cd.to(tl.float64))