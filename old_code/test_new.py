# test_new.py  (extended reporting: interpolation / extrapolation / training-fit)
import time
import re
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from itertools import product


# ----------------- Model (must match training) -----------------
class WNet(nn.Module):
    def __init__(self, layer_num: int, hidden_size: int, out_dim: int, in_features: int, act: str = "gelu"):
        super().__init__()
        act = (act or "gelu").lower().strip()
        if act not in {"tanh", "gelu", "gelu_tanh", "silu"}:
            raise ValueError("act must be one of: tanh, gelu, gelu_tanh, silu")

        def make_act():
            if act == "tanh":
                return nn.Tanh()
            if act == "gelu":
                return nn.GELU()
            if act == "gelu_tanh":
                return nn.GELU(approximate="tanh")
            if act == "silu":
                return nn.SiLU()
            raise RuntimeError("unreachable")

        layers = []
        for i in range(layer_num - 1):
            layers.append(nn.Linear(in_features if i == 0 else hidden_size, hidden_size))
            layers.append(make_act())
        layers.append(nn.Linear(hidden_size, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ----------------- Scaling / Chebyshev -----------------
def scale_to_unit_torch(x: torch.Tensor, a: float, b: float):
    """map [a,b] -> [-1,1]"""
    return 2.0 * (x - a) / (b - a) - 1.0


def chebyshev_T(x: torch.Tensor, M: int):
    N = x.shape[0]
    T = torch.empty((N, M + 1), device=x.device, dtype=x.dtype)
    T[:, 0] = 1.0
    if M >= 1:
        T[:, 1] = x
    for n in range(2, M + 1):
        T[:, n] = 2.0 * x * T[:, n - 1] - T[:, n - 2]
    return T


def chebyshev_basis_3d_aniso(
    beta_u: torch.Tensor,
    nu_u: torch.Tensor,
    rho_u: torch.Tensor,
    M_beta: int,
    M_nu: int,
    M_rho: int,
):
    Tb = chebyshev_T(beta_u, M_beta)
    Tn = chebyshev_T(nu_u, M_nu)
    Tr = chebyshev_T(rho_u, M_rho)
    Phi = torch.einsum("bi,bj,bk->bijk", Tb, Tn, Tr).reshape(beta_u.shape[0], -1)
    return Phi


# ----------------- Polynomial basis (NEW) -----------------
def polynomial_basis_3d(beta, nu, rho, M_beta, M_nu, M_rho):
    """
    Monomial basis: [beta^i * nu^j * rho^k] for i=0..M_beta, j=0..M_nu, k=0..M_rho
    Input beta/nu/rho: shape (B,) or (B,1) tensors.
    Return Phi: (B, K)
    """
    if beta.dim() == 2:
        beta = beta.squeeze(1)
    if nu.dim() == 2:
        nu = nu.squeeze(1)
    if rho.dim() == 2:
        rho = rho.squeeze(1)

    Tb = torch.stack([beta ** i for i in range(M_beta + 1)], dim=1)  # (B, M_beta+1)
    Tn = torch.stack([nu ** j for j in range(M_nu + 1)], dim=1)      # (B, M_nu+1)
    Tr = torch.stack([rho ** k for k in range(M_rho + 1)], dim=1)    # (B, M_rho+1)

    Phi = torch.einsum("bi,bj,bk->bijk", Tb, Tn, Tr).reshape(beta.shape[0], -1)
    return Phi


# ----------------- Data loader -----------------
def case_path(data_dir: str, beta: float, nu: float, rho: float):
    return f"{data_dir}/cdr_gt_beta_{beta:.2f}_nu_{nu:.2f}_rho_{rho:.2f}.npz"


def load_case(data_dir: str, beta: float, nu: float, rho: float):
    data = np.load(case_path(data_dir, beta, nu, rho))
    X = data["x"]
    T = data["t"]
    sol = data["solution"]
    X_grid, T_grid = np.meshgrid(X, T, indexing="ij")
    coords = np.concatenate((X_grid.reshape(-1, 1), T_grid.reshape(-1, 1)), axis=1).astype(np.float32)
    sol_flat = sol.reshape(-1, 1).astype(np.float32)
    return coords, sol_flat


# ----------------- Robust readers / fallbacks -----------------
def _npz_get_str(npz, key: str, default=None):
    if key not in npz:
        return default
    v = npz[key]
    if isinstance(v, np.ndarray) and v.dtype.kind in {"U", "S"}:
        return str(v.tolist()[0]) if v.ndim >= 1 else str(v)
    if isinstance(v, np.ndarray) and v.dtype == object:
        try:
            return str(v.tolist()[0])
        except Exception:
            return default
    return default


def infer_layers_hidden_from_foldername(name: str):
    mL = re.search(r"(?:^|_)layers(\d+)(?:_|$)", name)
    if mL is None:
        mL = re.search(r"(?:^|_)lrs(\d+)(?:_|$)", name)
    mH = re.search(r"(?:^|_)hs(\d+)(?:_|$)", name)
    if not (mL and mH):
        return None
    return int(mL.group(1)), int(mH.group(1))


def infer_M_from_foldername(name: str):
    m = re.search(r"_Mb(\d+)_Mn(\d+)_Mr(\d+)", name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def infer_M_shared_from_foldername(name: str):
    m = re.search(r"(?:^|_)M(\d+)(?:_|$)", name)
    if not m:
        return None
    M = int(m.group(1))
    return M, M, M


def infer_input_mode_in_dim_from_foldername(name: str):
    m = re.search(r"_in(xtp|xt)(\d+)", name)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def infer_act_from_foldername(name: str):
    m = re.search(r"_act-([A-Za-z0-9_]+)", name)
    return m.group(1) if m else None


def strip_compile_prefix(state_dict: dict) -> dict:
    has_prefixed = any(k.startswith("_orig_mod.") for k in state_dict.keys())
    if not has_prefixed:
        return state_dict
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_sd[k[len("_orig_mod."):]] = v
        else:
            new_sd[k] = v
    return new_sd


def infer_in_dim_from_state_dict(state_dict: dict) -> int:
    if "net.0.weight" in state_dict:
        w = state_dict["net.0.weight"]
        return int(w.shape[1])
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and v.ndim == 2 and k.endswith(".weight"):
            return int(v.shape[1])
    raise ValueError("Cannot infer in_dim from state_dict.")


def _ckpt_get_any(checkpoint: dict, keys):
    for k in keys:
        if k in checkpoint and checkpoint[k] is not None:
            return checkpoint[k]
    return None


def infer_xtp8_feat_from_norm(norm) -> str | None:
    """
    Return 'log1p' if norm contains xtp_extra_feat that matches log1p set.
    Return 'nd' if it matches nd_* set.
    Return None if cannot infer.
    """
    if "xtp_extra_feat" not in norm:
        return None
    v = norm["xtp_extra_feat"]
    try:
        feats = [str(x) for x in v.tolist()]
    except Exception:
        return None

    s = " ".join(feats).lower()
    if "log1p" in s:
        return "log1p"
    if "nd_" in s or "ndbeta" in s or "nd_beta" in s:
        return "nd"
    return None


# ----------------- NEW: infer training ranges from filename -----------------
def infer_train_ranges_from_foldername(name: str):
    """
    Parse training ranges from folder name.

    Supported patterns:
      1) shared range:  "..._range1-5_..."  -> beta/nu/rho all in [1,5]
      2) per-axis ranges:
         "..._rb0-20_rn0-20_rr0-20_..." -> beta in [0,20], nu in [0,20], rho in [0,20]
         (also accepts negative and decimals)

    Returns:
      dict with keys: beta, nu, rho each as [min,max] floats; or None if not found.
    """
    s = str(name)
    num = r"(-?\d+(?:\.\d+)?)"

    # Per-axis first (more specific)
    mb = re.search(rf"(?:^|_)rb{num}-{num}(?:_|$)", s, flags=re.IGNORECASE)
    mn = re.search(rf"(?:^|_)rn{num}-{num}(?:_|$)", s, flags=re.IGNORECASE)
    mr = re.search(rf"(?:^|_)rr{num}-{num}(?:_|$)", s, flags=re.IGNORECASE)
    if mb and mn and mr:
        rb = [float(mb.group(1)), float(mb.group(2))]
        rn = [float(mn.group(1)), float(mn.group(2))]
        rr = [float(mr.group(1)), float(mr.group(2))]
        return {"beta": rb, "nu": rn, "rho": rr}

    # Shared range
    g = re.search(rf"(?:^|_)range{num}-{num}(?:_|$)", s, flags=re.IGNORECASE)
    if g:
        lo, hi = float(g.group(1)), float(g.group(2))
        return {"beta": [lo, hi], "nu": [lo, hi], "rho": [lo, hi]}

    return None


def _validate_range_pair(pair, name="range"):
    if pair is None or len(pair) != 2:
        raise ValueError(f"{name} must be length-2, got {pair}")
    lo, hi = float(pair[0]), float(pair[1])
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError(f"{name} has non-finite values: {pair}")
    if hi < lo:
        raise ValueError(f"{name} has hi < lo: {pair}")
    return [lo, hi]


# ----------------- Metrics helpers -----------------
def percentile(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x.astype(np.float64), q))


def format_triplet(p):
    return f"(beta={p[0]:.2f}, nu={p[1]:.2f}, rho={p[2]:.2f})"


def build_input(
    all_xt: np.ndarray,
    all_params: np.ndarray,
    xt_mean,
    xt_std,
    input_mode: str,
    in_dim: int,
    target_range_beta,
    target_range_nu,
    target_range_rho,
    xtp8_feat: str = "nd",
):
    all_xt_norm = (all_xt - xt_mean) / xt_std

    beta_np = all_params[:, 0:1]
    nu_np = all_params[:, 1:2]
    rho_np = all_params[:, 2:3]

    # scaled to [-1,1]
    beta_u_np = (2.0 * (beta_np - target_range_beta[0]) / (target_range_beta[1] - target_range_beta[0]) - 1.0).astype(
        np.float32
    )
    nu_u_np = (2.0 * (nu_np - target_range_nu[0]) / (target_range_nu[1] - target_range_nu[0]) - 1.0).astype(
        np.float32
    )
    rho_u_np = (2.0 * (rho_np - target_range_rho[0]) / (target_range_rho[1] - target_range_rho[0]) - 1.0).astype(
        np.float32
    )

    if input_mode == "xt":
        all_input = all_xt_norm.astype(np.float32)

    else:
        # xtp: decide whether we are in xtp5 or xtp8 by in_dim
        if int(in_dim) == 5:
            # xt (2) + p3 (3) = 5
            p3 = np.concatenate([beta_u_np, nu_u_np, rho_u_np], axis=1).astype(np.float32)
            all_input = np.concatenate([all_xt_norm, p3], axis=1).astype(np.float32)

        elif int(in_dim) == 8:
            eps = 1e-12
            mode = (xtp8_feat or "nd").lower().strip()

            if mode == "nd":
                # 原有 test_cheb 行为（保留）
                nd_beta = beta_np / (float(target_range_beta[1]) + eps)
                nd_nu = nu_np / (float(target_range_nu[1]) + eps)
                nd_rho = rho_np / (float(target_range_rho[1]) + eps)
                p6 = np.concatenate([beta_u_np, nu_u_np, rho_u_np, nd_beta, nd_nu, nd_rho], axis=1).astype(np.float32)

            elif mode == "log1p":
                # 适配 train_new
                f1 = np.log1p(beta_np / (nu_np + eps))
                f2 = np.log1p(rho_np / (nu_np + eps))
                f3 = np.log1p(rho_np / (beta_np + eps))
                p6 = np.concatenate([beta_u_np, nu_u_np, rho_u_np, f1, f2, f3], axis=1).astype(np.float32)

            else:
                raise ValueError(f"Unknown xtp8_feat='{xtp8_feat}'. Choose from ['nd','log1p'].")

            all_input = np.concatenate([all_xt_norm, p6], axis=1).astype(np.float32)

        else:
            raise ValueError(
                f"Unsupported xtp in_dim={in_dim}. Expected 5 or 8. "
                f"(input_mode={input_mode})"
            )

    if all_input.shape[1] != int(in_dim):
        raise ValueError(
            f"Built input dim={all_input.shape[1]} but checkpoint expects in_dim={in_dim} (mode={input_mode})."
        )
    return all_input


def _make_test_values(min_val: float, step: float, max_val: float):
    """Build testing values from user-specified min and step, capped by max_val (inclusive)."""
    if step <= 0:
        raise ValueError(f"--param_step must be > 0, got {step}")
    if min_val > max_val + 1e-12:
        raise ValueError(f"param_min={min_val} is > max_val={max_val}")
    base = np.arange(float(min_val), float(max_val) + 1e-9, float(step), dtype=np.float32)
    return base.tolist()


@torch.no_grad()
def eval_cases(
    combos,
    data_dir,
    P,
    indices,
    xt_mean,
    xt_std,
    input_mode,
    in_dim,
    xtp8_feat,
    target_range_beta,
    target_range_nu,
    target_range_rho,
    netW,
    device,
    ftype,
    M_beta,
    M_nu,
    M_rho,
    basis: str = "cheb",
    clamp_size: float = 20.0,
):
    per_case = []
    skipped = []

    total_diff2 = 0.0
    total_sol2 = 0.0

    basis = (basis or "cheb").lower().strip()

    for (beta, nu, rho) in combos:
        pth = case_path(data_dir, float(beta), float(nu), float(rho))
        if not os.path.isfile(pth):
            skipped.append((float(beta), float(nu), float(rho), "missing_npz"))
            continue

        coords0, sol0 = load_case(data_dir, float(beta), float(nu), float(rho))
        N_total = coords0.shape[0]
        use_idx = indices[indices < N_total]
        if use_idx.shape[0] == 0:
            skipped.append((float(beta), float(nu), float(rho), "no_valid_indices"))
            continue
        use_idx = use_idx[: min(P, use_idx.shape[0])]

        xt = coords0[use_idx]
        u = sol0[use_idx]
        params = np.concatenate(
            [
                np.full((xt.shape[0], 1), float(beta), dtype=np.float32),
                np.full((xt.shape[0], 1), float(nu), dtype=np.float32),
                np.full((xt.shape[0], 1), float(rho), dtype=np.float32),
            ],
            axis=1,
        )

        all_input = build_input(
            all_xt=xt,
            all_params=params,
            xt_mean=xt_mean,
            xt_std=xt_std,
            input_mode=input_mode,
            in_dim=in_dim,
            target_range_beta=target_range_beta,
            target_range_nu=target_range_nu,
            target_range_rho=target_range_rho,
            xtp8_feat=xtp8_feat,
        )

        input_tensor = torch.from_numpy(all_input).to(device=device, dtype=ftype)
        u_tensor = torch.from_numpy(u).to(device=device, dtype=ftype)
        params_tensor = torch.from_numpy(params).to(device=device, dtype=ftype)

        beta_t = params_tensor[:, 0]
        nu_t = params_tensor[:, 1]
        rho_t = params_tensor[:, 2]

        # --- Phi by basis ---
        if basis == "cheb":
            beta_u = scale_to_unit_torch(beta_t, target_range_beta[0], target_range_beta[1])
            nu_u = scale_to_unit_torch(nu_t, target_range_nu[0], target_range_nu[1])
            rho_u = scale_to_unit_torch(rho_t, target_range_rho[0], target_range_rho[1])
            Phi = chebyshev_basis_3d_aniso(beta_u, nu_u, rho_u, M_beta, M_nu, M_rho)
        elif basis == "polyn":
            Phi = polynomial_basis_3d(beta_t, nu_t, rho_t, M_beta, M_nu, M_rho)
            Phi = torch.clamp(Phi, min=-float(clamp_size), max=float(clamp_size))
        else:
            raise ValueError(f"Unknown basis='{basis}'. Expected 'cheb' or 'polyn'.")

        W = netW(input_tensor)
        pred_u = torch.sum(W * Phi, dim=1, keepdim=True)

        diff_norm = torch.linalg.norm(pred_u - u_tensor, ord=2).item()
        sol_norm = torch.linalg.norm(u_tensor, ord=2).item()
        l2re = diff_norm / (sol_norm + 1e-12)

        per_case.append({"beta": float(beta), "nu": float(nu), "rho": float(rho), "l2re": float(l2re)})

        total_diff2 += diff_norm**2
        total_sol2 += sol_norm**2

    overall_l2re = (np.sqrt(total_diff2) / (np.sqrt(total_sol2) + 1e-12)) if total_sol2 > 0 else float("nan")
    return float(overall_l2re), per_case, skipped


def summarize_casewise_l2re(per_case: list[dict]):
    """Return case-wise mean/std/median/p95/max over per_case[*]['l2re']."""  # noqa: D401
    l2 = np.array([d["l2re"] for d in per_case], dtype=np.float64)
    if l2.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
            "max_triplet": (float("nan"), float("nan"), float("nan")),
        }

    imax = int(np.argmax(l2))
    mx_p = per_case[imax]
    return {
        "count": int(l2.size),
        "mean": float(np.mean(l2)),
        "std": float(np.std(l2)),
        "median": float(np.median(l2)),
        "p95": percentile(l2, 95.0),
        "max": float(l2[imax]),
        "max_triplet": (mx_p["beta"], mx_p["nu"], mx_p["rho"]),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--seed", type=int, default=6)

    parser.add_argument("--root_dir", type=str, default="CDR_using_new-config")
    parser.add_argument("--filename", type=str, required=True, help="<root_dir>/<filename>/checkpoint.pth")
    parser.add_argument("--data_dir", type=str, default="CDR_ground_truth_precise")

    parser.add_argument("--test_points", type=int, default=1024)

    # test parameter grid control (interpolation test set)
    parser.add_argument(
        "--param_min",
        type=float,
        default=1.25,
        help="Minimum TEST parameter value (used for interpolation testing grid), NOT training min.",
    )
    parser.add_argument("--param_step", type=float, default=0.5, help="TEST parameter spacing (used for interpolation testing grid).")

    # training grid evaluation spacing
    parser.add_argument("--train_step", type=float, default=0.5, help="TRAIN parameter spacing (used for training-point fit grid).")

    parser.add_argument(
        "--xtp8_feat",
        type=str,
        default="log1p",
        choices=["auto", "nd", "log1p"],
        help=(
            "Which 8D xtp feature set to use when in_dim==8. "
            "'nd' keeps the original test_cheb behavior: [beta_u,nu_u,rho_u, beta/hi, nu/hi, rho/hi]. "
            "'log1p' matches train_new: [beta_u,nu_u,rho_u, log1p(beta/nu), log1p(rho/nu), log1p(rho/beta)]. "
            "'auto' tries to infer from normalization_params.npz (xtp_extra_feat) or falls back to 'nd'."
        ),
    )

    parser.add_argument("--real_solution_size", type=int, default=-1, help="If <0, inferred from loaded case.")
    args = parser.parse_args()

    ftype = torch.float32
    torch.set_default_dtype(ftype)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    exp_dir = os.path.join(args.root_dir, args.filename)
    ckpt_path = os.path.join(exp_dir, "checkpoint.pth")
    norm_path = os.path.join(exp_dir, "normalization_params.npz")

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    if not os.path.isfile(norm_path):
        raise FileNotFoundError(f"normalization_params.npz not found: {norm_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_state_dict = checkpoint["model_state_dict"]
    state_dict = strip_compile_prefix(raw_state_dict)

    norm = np.load(norm_path, allow_pickle=True)

    xt_mean = norm["xt_mean"].astype(np.float32)
    xt_std = norm["xt_std"].astype(np.float32)
    xt_std[xt_std < 1e-8] = 1.0

    # --- basis / clamp_size (auto) ---
    basis = _npz_get_str(norm, "basis", default=None)
    if basis is None:
        basis = checkpoint.get("basis", None)
    basis = (basis or "cheb").lower().strip()
    if basis not in {"cheb", "polyn"}:
        print(f"[Warn] Unknown basis='{basis}' from files, fallback to 'cheb'.")
        basis = "cheb"

    clamp_size = None
    if "clamp_size" in norm:
        try:
            clamp_size = float(norm["clamp_size"])
        except Exception:
            clamp_size = None
    if clamp_size is None:
        v = checkpoint.get("clamp_size", None)
        clamp_size = float(v) if v is not None else 20.0

    # Prefer target_range_* for Chebyshev scaling
    if "target_range_beta" in norm and "target_range_nu" in norm and "target_range_rho" in norm:
        target_range_beta = norm["target_range_beta"].astype(np.float32).tolist()
        target_range_nu = norm["target_range_nu"].astype(np.float32).tolist()
        target_range_rho = norm["target_range_rho"].astype(np.float32).tolist()
    else:
        target_range_beta = norm["range_beta"].astype(np.float32).tolist()
        target_range_nu = norm["range_nu"].astype(np.float32).tolist()
        target_range_rho = norm["range_rho"].astype(np.float32).tolist()

    # Read / infer orders
    if "M_beta" in norm and "M_nu" in norm and "M_rho" in norm:
        M_beta = int(norm["M_beta"])
        M_nu = int(norm["M_nu"])
        M_rho = int(norm["M_rho"])
    else:
        parsed = infer_M_from_foldername(args.filename)
        if parsed is None:
            parsed = infer_M_shared_from_foldername(args.filename)
        if parsed is None:
            raise ValueError(
                "Cannot read M_beta/M_nu/M_rho from normalization_params or folder name. "
                "Need either npz keys M_beta/M_nu/M_rho, or name pattern _Mb#_Mn#_Mr#, or shared _M#."
            )
        M_beta, M_nu, M_rho = parsed

    K = (M_beta + 1) * (M_nu + 1) * (M_rho + 1)

    input_mode = _npz_get_str(norm, "input_mode", default=None)
    if input_mode is None:
        input_mode = checkpoint.get("input_mode", None)
    if input_mode is None:
        inferred = infer_input_mode_in_dim_from_foldername(args.filename)
        input_mode = inferred[0] if inferred else None

    in_dim = _ckpt_get_any(checkpoint, ["in_dim", "input_dim", "d_in"])
    if in_dim is None:
        inferred = infer_input_mode_in_dim_from_foldername(args.filename)
        in_dim = inferred[1] if inferred else None
    if in_dim is None:
        in_dim = infer_in_dim_from_state_dict(state_dict)
    in_dim = int(in_dim)

    if input_mode not in {"xt", "xtp"}:
        input_mode = "xt" if in_dim == 2 else "xtp"

    # ---- decide xtp8 feature set (only matters when input_mode=xtp and in_dim=8) ----
    xtp8_feat = args.xtp8_feat
    if input_mode == "xtp" and int(in_dim) == 8:
        if xtp8_feat == "auto":
            xtp8_feat = infer_xtp8_feat_from_norm(norm) or "nd"
    else:
        xtp8_feat = "nd"

    act = _npz_get_str(norm, "act", default=None)
    if act is None:
        act = _ckpt_get_any(checkpoint, ["act", "activation"])
    if act is None:
        act = infer_act_from_foldername(args.filename) or "gelu"

    layers = _ckpt_get_any(checkpoint, ["layers", "layer_num", "n_layers", "num_layers", "depth"])
    hidden = _ckpt_get_any(checkpoint, ["hidden_size", "hidden", "width", "hs", "d_hidden"])

    if layers is None or hidden is None:
        inferred = infer_layers_hidden_from_foldername(args.filename)
        if inferred is None:
            raise ValueError(
                "Cannot infer layers/hidden_size from checkpoint; and folder name missing patterns "
                "(_layers# or lrs#) and (_hs# or hs#)."
            )
        layers, hidden = inferred

    layers = int(layers)
    hidden = int(hidden)

    print(f"[Settings] input_mode={input_mode}, in_dim={in_dim}, act={act}, M=({M_beta},{M_nu},{M_rho}), K={K}")
    print(f"[Settings] basis={basis}, clamp_size={clamp_size} (only used when basis=polyn)")
    print(f"[Settings] target_range_beta/nu/rho = {target_range_beta}, {target_range_nu}, {target_range_rho}")
    print(f"[Settings] net layers={layers}, hidden={hidden}")
    print(f"[TestGrid] test_param_min={args.param_min}, test_param_step={args.param_step}")
    print(f"[TrainGrid] train_step={args.train_step} (train range inferred from filename if available)")

    netW = WNet(layer_num=layers, hidden_size=hidden, out_dim=int(K), in_features=in_dim, act=act)
    netW.load_state_dict(state_dict, strict=True)
    netW.to(device)
    netW.eval()

    # --- Prepare shared indices (same xt positions for fair comparisons) ---
    probe_candidates = [
        (float(target_range_beta[0]), float(target_range_nu[0]), float(target_range_rho[0])),
        (float(np.ceil(target_range_beta[0])), float(np.ceil(target_range_nu[0])), float(np.ceil(target_range_rho[0]))),
    ]
    coords0 = None
    N_total = None
    for pb, pn, pr in probe_candidates:
        pth = case_path(args.data_dir, float(pb), float(pn), float(pr))
        if os.path.isfile(pth):
            coords0, _ = load_case(args.data_dir, float(pb), float(pn), float(pr))
            N_total = coords0.shape[0]
            break
    if coords0 is None:
        raise FileNotFoundError("Cannot find any probe case to infer grid size. Check data_dir and file naming.")

    total_N = N_total if args.real_solution_size < 0 else int(args.real_solution_size)
    total_N = min(total_N, N_total)

    P = min(int(args.test_points), total_N)
    indices = np.random.choice(np.arange(total_N), size=P, replace=False)

    # ----------------- Interpolation combos (test grid) -----------------
    max_range = max(float(target_range_beta[1]), float(target_range_nu[1]), float(target_range_rho[1]))
    train_u_int = int(np.floor(max_range))  # 向下取整

    # Cap inside training upper bound to avoid boundary issues
    interp_cap = float(train_u_int) - float(args.param_step) / 2.0
    interp_cap = min(interp_cap, float(max_range))

    interp_vals = _make_test_values(min_val=float(args.param_min), step=float(args.param_step), max_val=float(interp_cap))
    interp_combos = list(product(interp_vals, interp_vals, interp_vals))
    interp_total = len(interp_combos)

    # ----------------- Reporting block -----------------
    print("\n==================== Evaluation Summary ====================")

    # 1) Interpolation ability
    t0 = time.time()
    overall_l2re_interp, per_case_interp, skipped_interp = eval_cases(
        combos=interp_combos,
        data_dir=args.data_dir,
        P=P,
        indices=indices,
        xt_mean=xt_mean,
        xt_std=xt_std,
        input_mode=input_mode,
        in_dim=in_dim,
        xtp8_feat=xtp8_feat,
        target_range_beta=target_range_beta,
        target_range_nu=target_range_nu,
        target_range_rho=target_range_rho,
        netW=netW,
        device=device,
        ftype=ftype,
        M_beta=M_beta,
        M_nu=M_nu,
        M_rho=M_rho,
        basis=basis,
        clamp_size=clamp_size,
    )
    sum_interp = summarize_casewise_l2re(per_case_interp)
    t1 = time.time()

    print("\n[1] Interpolation (off-grid) ability")
    print(f"  Interp values count={len(interp_vals)} (min={min(interp_vals):.3f}, max={max(interp_vals):.3f}), step={args.param_step}")
    print(f"  Interp combos total={interp_total}")
    print(f"  (evaluated={len(per_case_interp)}, skipped={len(skipped_interp)})")
    print(f"  Eval wall time (case-by-case): {t1 - t0:.6f} s")
    print(f"  Overall L2re (concat across interp cases): {overall_l2re_interp:.6e}  ({overall_l2re_interp*100:.4f}%)")
    print(
        "  Case-wise L2re stats over evaluated cases:"
        f" mean={sum_interp['mean']:.6e} ({sum_interp['mean']*100:.4f}%),"
        f" std={sum_interp['std']:.6e},"
        f" median={sum_interp['median']:.6e} ({sum_interp['median']*100:.4f}%),"
        f" P95={sum_interp['p95']:.6e} ({sum_interp['p95']*100:.4f}%),"
        f" max={sum_interp['max']:.6e} ({sum_interp['max']*100:.4f}%) at {format_triplet(sum_interp['max_triplet'])}"
    )
    if skipped_interp:
        miss = [p for p in skipped_interp if p[3] == "missing_npz"]
        if miss:
            print(f"  Skipped (missing npz): {len(miss)}. Example: {format_triplet((miss[0][0], miss[0][1], miss[0][2]))}")

    # 2) Extrapolation ability
    min_range = min(float(target_range_beta[0]), float(target_range_nu[0]), float(target_range_rho[0]))
    a = int(np.floor(max_range))
    b = int(np.floor((min_range + max_range) / 2.0))
    extra_combos = []
    for k in [1, 2, 3]:
        extra_combos.append((float(a + k), float(b), float(b)))
        extra_combos.append((float(b), float(a + k), float(b)))
        extra_combos.append((float(b), float(b), float(a + k)))

    overall_l2re_extra, per_case_extra, skipped_extra = eval_cases(
        combos=extra_combos,
        data_dir=args.data_dir,
        P=P,
        indices=indices,
        xt_mean=xt_mean,
        xt_std=xt_std,
        input_mode=input_mode,
        in_dim=in_dim,
        xtp8_feat=xtp8_feat,
        target_range_beta=target_range_beta,
        target_range_nu=target_range_nu,
        target_range_rho=target_range_rho,
        netW=netW,
        device=device,
        ftype=ftype,
        M_beta=M_beta,
        M_nu=M_nu,
        M_rho=M_rho,
        basis=basis,
        clamp_size=clamp_size,
    )

    print("\n[2] Extrapolation ability")
    print(f"  a=floor(max_range)={a}, b=floor((min+max)/2)={b}, test_points_per_case={P}")
    print("  Extrapolation points (9):")
    for p in extra_combos:
        print(f"    {format_triplet(p)}")
    if len(per_case_extra) > 0:
        for d in per_case_extra:
            p = (d["beta"], d["nu"], d["rho"])
            v = d["l2re"]
            print(f"  L2re {format_triplet(p)}: {v:.6e} ({v*100:.4f}%)")
        print(f"  Overall L2re (concat across available extrap cases): {overall_l2re_extra:.6e} ({overall_l2re_extra*100:.4f}%)")
    else:
        print("  No extrapolation cases evaluated (likely missing GT files).")
    if skipped_extra:
        miss = [p for p in skipped_extra if p[3] == "missing_npz"]
        if miss:
            print(f"  Skipped (missing npz): {len(miss)}. Example: {format_triplet((miss[0][0], miss[0][1], miss[0][2]))}")

    # 3) Training points (training grid) error
    inferred = infer_train_ranges_from_foldername(args.filename)

    fallback_beta = [float(target_range_beta[0]), float(target_range_beta[1])]
    fallback_nu = [float(target_range_nu[0]), float(target_range_nu[1])]
    fallback_rho = [float(target_range_rho[0]), float(target_range_rho[1])]

    if inferred is None:
        train_range_beta = fallback_beta
        train_range_nu = fallback_nu
        train_range_rho = fallback_rho
        used_filename_range = False
    else:
        train_range_beta = _validate_range_pair(inferred["beta"], "train_range_beta(from filename)")
        train_range_nu = _validate_range_pair(inferred["nu"], "train_range_nu(from filename)")
        train_range_rho = _validate_range_pair(inferred["rho"], "train_range_rho(from filename)")
        used_filename_range = True

    if args.train_step <= 0:
        raise ValueError(f"--train_step must be > 0, got {args.train_step}")

    beta_vals = np.arange(train_range_beta[0], train_range_beta[1] + 1e-9, float(args.train_step), dtype=np.float32)
    nu_vals = np.arange(train_range_nu[0], train_range_nu[1] + 1e-9, float(args.train_step), dtype=np.float32)
    rho_vals = np.arange(train_range_rho[0], train_range_rho[1] + 1e-9, float(args.train_step), dtype=np.float32)

    train_combos = list(product(beta_vals.tolist(), nu_vals.tolist(), rho_vals.tolist()))

    overall_l2re_train, per_case_train, skipped_train = eval_cases(
        combos=train_combos,
        data_dir=args.data_dir,
        P=P,
        indices=indices,
        xt_mean=xt_mean,
        xt_std=xt_std,
        input_mode=input_mode,
        in_dim=in_dim,
        xtp8_feat=xtp8_feat,
        target_range_beta=target_range_beta,
        target_range_nu=target_range_nu,
        target_range_rho=target_range_rho,
        netW=netW,
        device=device,
        ftype=ftype,
        M_beta=M_beta,
        M_nu=M_nu,
        M_rho=M_rho,
        basis=basis,
        clamp_size=clamp_size,
    )
    sum_train = summarize_casewise_l2re(per_case_train)

    print("\n[3] Training-point fit (training grid)")
    print(f"  Train ranges: beta={train_range_beta}, nu={train_range_nu}, rho={train_range_rho}")
    print(f"  Train step={args.train_step} => combos={len(train_combos)} (evaluated={len(per_case_train)}, skipped={len(skipped_train)})")
    if used_filename_range:
        print(f"  Note: training ranges inferred from filename='{args.filename}'")
    else:
        print("  Note: training ranges NOT found in filename; used normalization ranges as fallback.")
    print(f"  Overall L2re (concat across train cases): {overall_l2re_train:.6e}  ({overall_l2re_train*100:.4f}%)")
    print(
        "  Case-wise L2re stats over evaluated cases:"
        f" mean={sum_train['mean']:.6e} ({sum_train['mean']*100:.4f}%),"
        f" std={sum_train['std']:.6e},"
        f" median={sum_train['median']:.6e} ({sum_train['median']*100:.4f}%),"
        f" P95={sum_train['p95']:.6e} ({sum_train['p95']*100:.4f}%),"
        f" max={sum_train['max']:.6e} ({sum_train['max']*100:.4f}%) at {format_triplet(sum_train['max_triplet'])}"
    )
    if skipped_train:
        miss = [p for p in skipped_train if p[3] == "missing_npz"]
        if miss:
            print(f"  Skipped (missing npz): {len(miss)}. Example: {format_triplet((miss[0][0], miss[0][1], miss[0][2]))}")

    print("\n============================================================")


if __name__ == "__main__":
    main()