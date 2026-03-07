from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


def scale_to_unit_torch(x: torch.Tensor, a: float, b: float) -> torch.Tensor:
    return 2.0 * (x - a) / (b - a) - 1.0


def scale_to_unit_np(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return 2.0 * (x - a) / (b - a) - 1.0


def chebyshev_T(x: torch.Tensor, M: int) -> torch.Tensor:
    """
    x: (B,) tensor in [-1,1]
    returns: (B, M+1)
    """
    if x.dim() != 1:
        x = x.reshape(-1)
    B = x.shape[0]
    T = torch.empty((B, M + 1), device=x.device, dtype=x.dtype)
    T[:, 0] = 1.0
    if M >= 1:
        T[:, 1] = x
    for n in range(2, M + 1):
        T[:, n] = 2.0 * x * T[:, n - 1] - T[:, n - 2]
    return T


def chebyshev_T_np(x: np.ndarray, M: int) -> np.ndarray:
    x = x.astype(np.float64, copy=False).reshape(-1)
    B = x.shape[0]
    T = np.empty((B, M + 1), dtype=np.float64)
    T[:, 0] = 1.0
    if M >= 1:
        T[:, 1] = x
    for n in range(2, M + 1):
        T[:, n] = 2.0 * x * T[:, n - 1] - T[:, n - 2]
    return T


def _poly_powers_1d(p: torch.Tensor, M: int) -> torch.Tensor:
    if p.dim() != 1:
        p = p.reshape(-1)
    return torch.stack([p**i for i in range(M + 1)], dim=1)


def chebyshev_basis_nd(params_unit: Sequence[torch.Tensor], Ms: Sequence[int]) -> torch.Tensor:
    """
    params_unit: list of (B,) tensors, each in [-1,1]
    Ms: list of degrees [M1, M2, ...]
    return: Phi (B, Π_i (Mi+1))
    """
    if len(params_unit) != len(Ms):
        raise ValueError("len(params_unit) must equal len(Ms)")
    if len(Ms) < 1 or len(Ms) > 3:
        raise ValueError("Only param_dim in {1,2,3} is supported")

    Ts = [chebyshev_T(p, int(M)) for p, M in zip(params_unit, Ms)]

    if len(Ts) == 1:
        return Ts[0]
    if len(Ts) == 2:
        return torch.einsum("bi,bj->bij", Ts[0], Ts[1]).reshape(Ts[0].shape[0], -1)
    return torch.einsum("bi,bj,bk->bijk", Ts[0], Ts[1], Ts[2]).reshape(Ts[0].shape[0], -1)


def polynomial_basis_nd(params_unit: Sequence[torch.Tensor], Ms: Sequence[int]) -> torch.Tensor:
    """
    Monomial basis on provided parameters.
    NOTE: Despite the name params_unit, the input can be raw params as well.
    params_unit: list of (B,) tensors
    Ms: list of degrees
    return: Phi (B, Π_i (Mi+1))
    """
    if len(params_unit) != len(Ms):
        raise ValueError("len(params_unit) must equal len(Ms)")
    if len(Ms) < 1 or len(Ms) > 3:
        raise ValueError("Only param_dim in {1,2,3} is supported")

    Ts = [_poly_powers_1d(p, int(M)) for p, M in zip(params_unit, Ms)]

    if len(Ts) == 1:
        return Ts[0]
    if len(Ts) == 2:
        return torch.einsum("bi,bj->bij", Ts[0], Ts[1]).reshape(Ts[0].shape[0], -1)
    return torch.einsum("bi,bj,bk->bijk", Ts[0], Ts[1], Ts[2]).reshape(Ts[0].shape[0], -1)


def compute_K(Ms: Sequence[int]) -> int:
    K = 1
    for M in Ms:
        K *= (int(M) + 1)
    return int(K)


@dataclass(frozen=True)
class BasisConfig:
    kind: str  # "cheb" or "polyn"


def build_param_basis(
    *,
    basis_cfg: BasisConfig,
    params_raw: Sequence[torch.Tensor],
    params_unit: Sequence[torch.Tensor],
    Ms: Sequence[int],
    polyn_use_raw: bool = False,
) -> torch.Tensor:
    """
    Unified basis builder for param_dim in {1,2,3}.
    - For cheb: uses params_unit (scaled to [-1,1])
    - For polyn:
        - if polyn_use_raw=True: uses params_raw (no scaling)
        - else: uses params_unit (scaled to [-1,1])  # original behavior
    """
    kind = basis_cfg.kind.lower().strip()
    if kind == "cheb":
        Phi = chebyshev_basis_nd(params_unit, Ms)
    elif kind == "polyn":
        if polyn_use_raw:
            Phi = polynomial_basis_nd(params_raw, Ms)
        else:
            Phi = polynomial_basis_nd(params_unit, Ms)
    else:
        raise ValueError("basis_cfg.kind must be 'cheb' or 'polyn'")
    return Phi.contiguous()