from __future__ import annotations

import os
from typing import Tuple

import numpy as np

# Optional deps used by some PDEs
import pandas as pd
from scipy.io import loadmat


def make_grid(a: float, b: float, step: float, decimals: int = 2) -> np.ndarray:
    if step <= 0:
        raise ValueError("step must be > 0")
    n = int(round((b - a) / step))
    vals = [round(a + i * step, decimals) for i in range(n + 1)]
    vals = [v for v in vals if v <= b + 1e-12]
    vals = sorted(list(dict.fromkeys(vals)))
    return np.array(vals, dtype=np.float32)


def _join(data_dir: str, fname: str) -> str:
    return os.path.join(data_dir, fname)


def _as_2d_grid_coords_and_u(X, T, U) -> tuple[np.ndarray, np.ndarray]:
    """
    Supports:
      A) x,t,u are 2D grids (same shape)
      B) x,t 1D axes and u 2D (Nx,Nt) or (Nt,Nx)

    Returns:
      coords: (N,2)
      sol_flat: (N,1)
    """
    X = np.asarray(X)
    T = np.asarray(T)
    U = np.asarray(U)

    if X.ndim == 2 and T.ndim == 2:
        if U.ndim != 2:
            raise ValueError(f"u must be 2D when x/t are 2D grids (got {U.shape})")
        if X.shape != T.shape or X.shape != U.shape:
            raise ValueError(
                "grid-shape mismatch:\n"
                f"x.shape={X.shape}, t.shape={T.shape}, u.shape={U.shape}\n"
                "Expected all equal when x/t are stored as 2D grids."
            )
        coords = np.concatenate([X.reshape(-1, 1), T.reshape(-1, 1)], axis=1).astype(np.float32)
        sol_flat = U.reshape(-1, 1).astype(np.float32)
        return coords, sol_flat

    if X.ndim == 1 and T.ndim == 1:
        if U.ndim != 2:
            raise ValueError(f"u must be 2D when x/t are 1D axes (got {U.shape})")
        Nx, Nt = int(X.shape[0]), int(T.shape[0])
        if U.shape == (Nx, Nt):
            U_xt = U
        elif U.shape == (Nt, Nx):
            U_xt = U.T
        else:
            raise ValueError(
                "u shape mismatch for 1D axes:\n"
                f"len(x)={Nx}, len(t)={Nt}, u.shape={U.shape}\n"
                "Expected (Nx,Nt) or (Nt,Nx)."
            )
        X_grid, T_grid = np.meshgrid(X, T, indexing="ij")
        coords = np.concatenate([X_grid.reshape(-1, 1), T_grid.reshape(-1, 1)], axis=1).astype(np.float32)
        sol_flat = U_xt.reshape(-1, 1).astype(np.float32)
        return coords, sol_flat

    raise ValueError(
        "Unsupported format:\n"
        f"x.ndim={X.ndim}, t.ndim={T.ndim}, u.ndim={U.ndim}\n"
        "Expected either (x,t,u) all 2D grids or (x,t) 1D axes with u 2D."
    )


# ----------------- CDR-specific data helpers (implemented) -----------------
def case_filename_cdr(data_dir: str, beta: float, nu: float, rho: float) -> str:
    return _join(data_dir, f"cdr_gt_beta_{beta:.2f}_nu_{nu:.2f}_rho_{rho:.2f}.npz")


def load_case_cdr(data_dir: str, beta: float, nu: float, rho: float):
    data = np.load(case_filename_cdr(data_dir, beta, nu, rho))
    X = data["x"]
    T = data["t"]
    sol = data["solution"]
    X_grid, T_grid = np.meshgrid(X, T, indexing="ij")
    coords = np.concatenate((X_grid.reshape(-1, 1), T_grid.reshape(-1, 1)), axis=1).astype(np.float32)
    sol_flat = sol.reshape(-1, 1).astype(np.float32)
    return coords, sol_flat


def load_solution_only_cdr(data_dir: str, beta: float, nu: float, rho: float):
    data = np.load(case_filename_cdr(data_dir, beta, nu, rho))
    sol = data["solution"].astype(np.float32)
    return sol.reshape(-1, 1)


# ----------------- Convection (implemented using your GT format) -----------------
def case_filename_convection(data_dir: str, beta: float) -> str:
    return _join(data_dir, f"beta={beta:.2f}_conv_ground_truth.npz")


def load_case_convection(data_dir: str, beta: float):
    data = np.load(case_filename_convection(data_dir, beta))
    X = data["x"]
    T = data["t"]
    U = data["u_true"]
    coords, sol_flat = _as_2d_grid_coords_and_u(X, T, U)
    return coords, sol_flat


def load_solution_only_convection(data_dir: str, beta: float):
    data = np.load(case_filename_convection(data_dir, beta))
    X = data["x"]
    T = data["t"]
    U = data["u_true"]
    _, sol_flat = _as_2d_grid_coords_and_u(X, T, U)
    return sol_flat


# ----------------- Heat (xt, alpha) -----------------
def case_filename_heat(data_dir: str, alpha: float) -> str:
    # Example: a=0.5_heat_eq_testdata.npz
    return _join(data_dir, f"a={alpha:.1f}_heat_eq_testdata.npz")


def load_case_heat(data_dir: str, alpha: float):
    """
    Provided format:
      data = np.load("realSolution/a={:.1f}_heat_eq_testdata.npz".format(alpha))
      solution = data["usol"]
    Assumption:
      - "usol" is a 2D array on a regular (x,t) grid.
      - If x/t axes are not stored, we use the benchmark domain x∈[0,2π], t∈[0,1]
        with sizes inferred from usol.shape.
    """
    data = np.load(case_filename_heat(data_dir, alpha))
    U = np.asarray(data["usol"])

    if U.ndim != 2:
        raise ValueError(f"Heat usol must be 2D (got {U.shape})")

    Nx, Nt = U.shape[0], U.shape[1]
    x = np.linspace(0.0, 2.0 * np.pi, Nx, endpoint=True, dtype=np.float32)
    t = np.linspace(0.0, 1.0, Nt, endpoint=True, dtype=np.float32)
    coords, sol_flat = _as_2d_grid_coords_and_u(x, t, U)
    return coords, sol_flat


def load_solution_only_heat(data_dir: str, alpha: float):
    _, sol = load_case_heat(data_dir, alpha)
    return sol


# ----------------- Reaction (xt, rho) -----------------
def case_filename_reaction(data_dir: str, rho: float) -> str:
    # Example: rho=1.00-real_solution.npz
    return _join(data_dir, f"rho={rho:.2f}-real_solution.npz")


def load_case_reaction(data_dir: str, rho: float):
    """
    Provided format:
      data = np.load("rho={:.2f}-real_solution.npz".format(rho))
      X = data["x"]; T = data["t"]; solution = data["solution"]
    X/T could be axes (1D) or grids (2D). solution should be 2D accordingly.
    """
    data = np.load(case_filename_reaction(data_dir, rho))
    X = data["x"]
    T = data["t"]
    U = data["solution"]
    coords, sol_flat = _as_2d_grid_coords_and_u(X, T, U)
    return coords, sol_flat


def load_solution_only_reaction(data_dir: str, rho: float):
    _, sol = load_case_reaction(data_dir, rho)
    return sol


# ----------------- Burgers (xt, nu) -----------------
def case_filename_burgers(data_dir: str, nu: float) -> str:
    # Example: nu=0.001_real_solution.csv ; nu=0.0015_real_solution.csv
    # Use 4 decimals to represent values like 0.0015 exactly in the name.
    return _join(data_dir, f"nu={nu:.4f}_real_solution.csv")


def _burgers_parse_csv_values(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    We try to support two plausible CSV layouts:

    Layout 1 (likely): columns are [x, u(t0), u(t1), ..., u(t_{Nt-1})]
      -> arr shape (Nx, Nt+1)

    Layout 2: pure u grid (Nx, Nt)
      -> arr shape (Nx, Nt) with no x column.

    Returns: (x, t, U_xt) where U_xt is (Nx, Nt)
    """
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Burgers CSV must be 2D (got {arr.shape})")

    Nx, C = arr.shape

    if C >= 2:
        # assume first col is x if it looks monotonic-ish
        x0 = arr[:, 0]
        if np.all(np.isfinite(x0)) and np.nanmin(x0) >= -2.0 and np.nanmax(x0) <= 2.0:
            # Heuristic: treat col0 as x
            x = x0.astype(np.float32)
            U = arr[:, 1:].astype(np.float32)
            Nt = U.shape[1]
            t = np.linspace(0.0, 1.0, Nt, endpoint=True, dtype=np.float32)
            return x, t, U

    # fallback: assume arr is U directly
    U = arr.astype(np.float32)
    Nx, Nt = U.shape
    x = np.linspace(-1.0, 1.0, Nx, endpoint=True, dtype=np.float32)
    t = np.linspace(0.0, 1.0, Nt, endpoint=True, dtype=np.float32)
    return x, t, U


def load_case_burgers(data_dir: str, nu: float):
    """
    Provided format:
      df = pd.read_csv("realSolution/nu={:.3f}_real_solution.csv".format(nu), header=None)
      data = df.values
    Note: your example uses {:.3f} but also mentions 0.0015 (needs 4 decimals).
    Here we use {:.4f} in filename; adjust if your files actually use 3 decimals only.
    """
    df = pd.read_csv(case_filename_burgers(data_dir, nu), header=None)
    arr = df.values
    x, t, U = _burgers_parse_csv_values(arr)
    coords, sol_flat = _as_2d_grid_coords_and_u(x, t, U)
    return coords, sol_flat


def load_solution_only_burgers(data_dir: str, nu: float):
    _, sol = load_case_burgers(data_dir, nu)
    return sol


# ----------------- Allen–Cahn 1D+T (xt, lambda, epsilon) -----------------
def case_filename_allen_cahn_1d(data_dir: str, lam: float, eps: float) -> str:
    # Example: test/lambda=0.00010,epsilon=1.0.csv
    return _join(data_dir, f"lambda={lam:.5f},epsilon={eps:.1f}.csv")


def load_case_allen_cahn_1d(data_dir: str, lam: float, eps: float):
    """
    Provided format:
      df = pd.read_csv("test/lambda={:.5f},epsilon={:.1f}.csv".format(lam, eps), header=None)
      all_sol_data = df.values
      x = all_sol_data[:,0]
      t = linspace(0,1, num=128)
    Assumption:
      - CSV contains x in first column, and solution values over time in remaining columns: (Nx, 1+Nt)
    """
    df = pd.read_csv(case_filename_allen_cahn_1d(data_dir, lam, eps), header=None)
    arr = df.values
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Allen-Cahn 1D CSV must be (Nx,1+Nt), got {arr.shape}")

    x = arr[:, 0].astype(np.float32)
    U = arr[:, 1:].astype(np.float32)  # (Nx,Nt)
    Nt = U.shape[1]
    t = np.linspace(0.0, 1.0, num=Nt, endpoint=True, dtype=np.float32)

    coords, sol_flat = _as_2d_grid_coords_and_u(x, t, U)
    return coords, sol_flat


def load_solution_only_allen_cahn_1d(data_dir: str, lam: float, eps: float):
    _, sol = load_case_allen_cahn_1d(data_dir, lam, eps)
    return sol


# ----------------- Allen–Cahn 2D+T (xyt, lambda, epsilon) -----------------
def case_filename_allen_cahn_2d(data_dir: str, lam: float, eps: float) -> str:
    # Example: lbd=0.00010,eps=1.0_ac2d.mat
    # Note the key name uses "lbd" and "eps"
    return _join(data_dir, f"lbd={lam:.5f},eps={eps:.1f}_ac2d.mat")


def load_case_allen_cahn_2d(data_dir: str, lam: float, eps: float):
    """
    Provided format:
      mat = loadmat(".../lbd={:.5f},eps={:.1f}_ac2d.mat".format(lam, eps))
      U = mat["u"]
    We need (x,y,t) coords. If axes are not provided, we assume benchmark domain:
      x,y in [-1,1], t in [0,1]
    and infer sizes from U shape.

    U expected to be (Nx,Ny,Nt) OR (Ny,Nx,Nt) etc.
    We'll assume (Nx,Ny,Nt). If your data uses a different order, adjust here.
    """
    mat = loadmat(case_filename_allen_cahn_2d(data_dir, lam, eps))
    if "u" not in mat:
        raise KeyError(f"Allen-Cahn 2D MAT file missing key 'u'. Keys: {list(mat.keys())}")

    U = np.asarray(mat["u"])
    if U.ndim != 3:
        raise ValueError(f"Allen-Cahn 2D u must be 3D (got {U.shape})")

    Nx, Ny, Nt = U.shape[0], U.shape[1], U.shape[2]
    x = np.linspace(-1.0, 1.0, Nx, endpoint=True, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, Ny, endpoint=True, dtype=np.float32)
    t = np.linspace(0.0, 1.0, Nt, endpoint=True, dtype=np.float32)

    # build coords (Nx*Ny*Nt, 3) with indexing ij
    Xg, Yg, Tg = np.meshgrid(x, y, t, indexing="ij")
    coords = np.stack([Xg.reshape(-1), Yg.reshape(-1), Tg.reshape(-1)], axis=1).astype(np.float32)
    sol_flat = U.reshape(-1, 1).astype(np.float32)
    return coords, sol_flat


def load_solution_only_allen_cahn_2d(data_dir: str, lam: float, eps: float):
    _, sol = load_case_allen_cahn_2d(data_dir, lam, eps)
    return sol


# ----------------- 2D+T Navier–Stokes (xyt, viscosity) -----------------
def case_filename_ns_2d(data_dir: str, viscosity: float) -> str:
    # Example: ns_data_v=0.00143.mat
    return _join(data_dir, f"ns_data_v={viscosity:.5f}.mat")


def load_case_ns_2d(data_dir: str, viscosity: float):
    """
    Minimal, generic loader (based on your snippet):
      data = loadmat("ns_data_v={:.4f}.mat".format(viscosity))
      u = data["u"][0]  # (Nx,Ny,Nt) typically
    We return coords on full grid and u flattened.

    If the file stores x/y/t axes, prefer using them; otherwise assume x,y,t in [0,1].
    """
    mat = loadmat(case_filename_ns_2d(data_dir, viscosity))
    if "u" not in mat:
        raise KeyError(f"NS MAT file missing key 'u'. Keys: {list(mat.keys())}")

    U = mat["u"]
    # Some MAT store u as object array with shape (1,1) or (1,)
    if isinstance(U, np.ndarray) and U.dtype == object:
        U = U.item()

    U = np.asarray(U)
    if U.ndim == 4 and U.shape[0] == 1:
        U = U[0]

    if U.ndim != 3:
        raise ValueError(f"NS u must be 3D (got {U.shape})")

    Nx, Ny, Nt = U.shape
    x = np.linspace(0.0, 1.0, Nx, endpoint=True, dtype=np.float32)
    y = np.linspace(0.0, 1.0, Ny, endpoint=True, dtype=np.float32)
    t = np.linspace(0.0, 1.0, Nt, endpoint=True, dtype=np.float32)

    Xg, Yg, Tg = np.meshgrid(x, y, t, indexing="ij")
    coords = np.stack([Xg.reshape(-1), Yg.reshape(-1), Tg.reshape(-1)], axis=1).astype(np.float32)
    sol_flat = U.reshape(-1, 1).astype(np.float32)
    return coords, sol_flat


def load_solution_only_ns_2d(data_dir: str, viscosity: float):
    _, sol = load_case_ns_2d(data_dir, viscosity)
    return sol