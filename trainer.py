from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm

from basis import (
    BasisConfig,
    build_param_basis,
    compute_K,
    scale_to_unit_torch,
)
from model import WNet


# ----------------- utils merged here -----------------
def set_seed(myseed: int):
    random.seed(myseed)
    os.environ["PYTHONHASHSEED"] = str(myseed)
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    torch.cuda.manual_seed(myseed)
    torch.cuda.manual_seed_all(myseed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _count_lr_updates_in_range(
    start_inclusive: int, end_exclusive: int, step: int, exclude_zero: bool = True
) -> int:
    if step <= 0:
        raise ValueError("step must be > 0")
    start_inclusive = int(start_inclusive)
    end_exclusive = int(end_exclusive)
    if end_exclusive <= start_inclusive:
        return 0

    first = ((start_inclusive + step - 1) // step) * step
    if exclude_zero and first == 0:
        first += step

    if first >= end_exclusive:
        return 0

    last = ((end_exclusive - 1) // step) * step
    if exclude_zero and last == 0:
        return 0

    if last < first:
        return 0
    return 1 + (last - first) // step


def _gamma_for_target_ratio(target_ratio: float, n_updates: int) -> float:
    n_updates = int(n_updates)
    if n_updates <= 0:
        return 1.0
    if target_ratio <= 0.0:
        raise ValueError("target_ratio must be > 0")
    return float(target_ratio ** (1.0 / n_updates))


# ----------------- Loss -----------------
def loss_mse_nmae(pred, truth, eps=1e-3, mix_lambda=0.95, per_case=False, Bcase=None, Pxt=None):
    if not per_case:
        mse = torch.mean((pred - truth) ** 2)
        nmae = torch.mean(torch.abs(pred - truth) / (torch.abs(truth) + eps))
        return mix_lambda * mse + (1.0 - mix_lambda) * nmae, mse.detach(), nmae.detach()

    assert Bcase is not None and Pxt is not None
    pred2 = pred.reshape(Bcase, Pxt)
    tru2 = truth.reshape(Bcase, Pxt)
    mse_case = torch.mean((pred2 - tru2) ** 2, dim=1)
    nmae_case = torch.mean(torch.abs(pred2 - tru2) / (torch.abs(tru2) + eps), dim=1)
    mse = torch.mean(mse_case)
    nmae = torch.mean(nmae_case)
    return mix_lambda * mse + (1.0 - mix_lambda) * nmae, mse.detach(), nmae.detach()


def _ranges_from_cases(cases: np.ndarray) -> Tuple[Tuple[float, float], ...]:
    if cases.ndim != 2:
        raise ValueError("cases must be 2D (num_cases, d)")
    d = cases.shape[1]
    out = []
    for i in range(d):
        col = cases[:, i].astype(np.float64)
        out.append((float(col.min()), float(col.max())))
    return tuple(out)


@dataclass
class TrainConfig:
    basis: str = "cheb"  # "cheb" or "polyn"

    layers: int = 6
    hidden_size: int = 192
    act: str = "gelu"

    Ms: Tuple[int, ...] = (6, 6, 6)

    pretrain: bool = False
    target_param_ranges: Optional[Tuple[Tuple[float, float], ...]] = None

    input_mode: str = "auto"  # "auto" | "xt" | "xtp"
    auto_threshold: float = 10.0
    use_ratio_feats: bool = False

    device: str = "0"
    seed: int = 12
    lr: float = 1e-3
    weight_decay: float = 0.0
    iters: int = 200000
    lr_gamma1: float = 0.99
    lr_gamma2: float = 0.987
    eval_interval: int = 500

    loss_mode: str = "mse"
    mix_lambda: float = 0.95
    nmae_eps: float = 1e-3
    per_case_loss: bool = False

    xt_points: int = 1000

    case_batch: int = 0
    resample_case_interval: int = 1000

    resample_xt: bool = True
    resample_xt_interval: int = 5000

    test_params: Tuple[float, ...] = (3.25, 3.25, 3.25)

    data_dir: str = ""
    save_dir: str = "runs"

    warm_start_path: str = ""
    warm_start_strict: bool = False

    U_all_on_gpu: int = 1

    amp80: bool = True

    # 1: standardized coords + normalized param feats for net input
    # 0: raw coords + raw params into net input (no ratio feats)
    # NOTE: Phi (cheb/polyn) always uses unit parameters in [-1,1].
    normalize_io: int = 1


def _build_param_features(
    params_raw_t: torch.Tensor,  # (Bcase, d)
    target_ranges: Sequence[Tuple[float, float]],
    use_ratio_feats: bool,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    d = params_raw_t.shape[1]
    units = []
    unit_names = []
    for i in range(d):
        a, b = target_ranges[i]
        ui = scale_to_unit_torch(params_raw_t[:, i], float(a), float(b))
        units.append(ui)
        unit_names.append(f"unit_p{i}")
    params_unit_t = torch.stack(units, dim=1).contiguous()

    if not use_ratio_feats or d < 2:
        return params_unit_t, params_unit_t, unit_names

    ratio_feats = []
    ratio_names = []
    for i in range(d):
        for j in range(i + 1, d):
            pi = params_raw_t[:, i : i + 1]
            pj = params_raw_t[:, j : j + 1]
            ratio_feats.append(torch.log1p(pi / (pj + eps)))
            ratio_names.append(f"log1p(p{i}/p{j})")

    ratio_t = torch.cat(ratio_feats, dim=1)
    feat = torch.cat([params_unit_t, ratio_t], dim=1).contiguous()
    names = unit_names + ratio_names
    return params_unit_t, feat, names


def _format_param_line(param_names: Sequence[str], params: Sequence[float]) -> str:
    parts = [f"{n}={float(v):.10g}" for n, v in zip(param_names, params)]
    return ", ".join(parts)


def _count_leading_zeros_after_decimal(x: float) -> int:
    """
    For 0 < x < 1:
    returns number of consecutive zeros right after decimal point
    e.g. x=0.0000123 -> 4
         x=0.00123   -> 2
         x=0.1       -> 0
    """
    if not np.isfinite(x) or x <= 0.0 or x >= 1.0:
        return 0
    e = int(np.floor(np.log10(x)))  # negative
    return max(0, -e - 1)


def _format_std(std: float) -> str:
    """
    Rule:
    - if std has >3 zeros after decimal (i.e., 0.0000x...), use scientific notation with 2 sig figs
    - else normal decimal, trimming trailing zeros
    """
    std = float(std)
    if not np.isfinite(std):
        return str(std)
    if std == 0.0:
        return "0"

    a = abs(std)
    if 0.0 < a < 1.0 and _count_leading_zeros_after_decimal(a) > 3:
        return f"{std:.2e}"

    s = f"{std:.12f}".rstrip("0").rstrip(".")
    return s if s else "0"


@torch.no_grad()
def final_test_after_training(
    spec,
    cfg: TrainConfig,
    folder_path: str,
    folder_name: str,
    device: torch.device,
    coord_mean: np.ndarray,
    coord_std: np.ndarray,
    target_ranges: Sequence[Tuple[float, float]],
    Ms: Sequence[int],
    input_mode: str,
    use_ratio_feats: bool,
    act: str,
):
    test_cases = np.asarray(getattr(spec, "final_test_param_cases", None))
    if test_cases is None or test_cases.size == 0:
        print("[Final Test] spec.final_test_param_cases is empty; skip.")
        return

    if test_cases.ndim != 2 or test_cases.shape[1] != spec.param_dim:
        raise ValueError(f"final_test_param_cases must be (N,{spec.param_dim}), got {test_cases.shape}")

    param_names = list(getattr(spec, "param_names", [f"p{i}" for i in range(spec.param_dim)]))
    if len(param_names) != spec.param_dim:
        param_names = [f"p{i}" for i in range(spec.param_dim)]

    ftype = torch.float32
    basis_cfg = BasisConfig(kind=cfg.basis)
    K = compute_K([int(m) for m in Ms])

    if input_mode == "xt":
        param_feat_dim = 0
    else:
        if cfg.normalize_io == 1:
            param_feat_dim = spec.param_dim + (
                (spec.param_dim * (spec.param_dim - 1)) // 2 if use_ratio_feats else 0
            )
        else:
            param_feat_dim = spec.param_dim
    in_dim = spec.coord_dim if input_mode == "xt" else (spec.coord_dim + param_feat_dim)

    ckpt_path = os.path.join(folder_path, "checkpoint.pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    netW = WNet(layer_num=cfg.layers, hidden_size=cfg.hidden_size, out_dim=K, in_features=in_dim, act=act)
    netW.initialize()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    netW.load_state_dict(ckpt["model_state_dict"], strict=True)
    netW.to(device)
    netW.eval()

    eps_den = 1e-12
    l2_list: List[float] = []

    lines: List[str] = []
    lines.append(f"folder: {folder_name}")

    for i in range(test_cases.shape[0]):
        params = tuple(float(x) for x in test_cases[i].tolist())

        coords, u_true = spec.load_case(cfg.data_dir, *params)
        if coords.shape[1] != spec.coord_dim:
            raise ValueError("coord_dim mismatch in final test")

        if cfg.normalize_io == 1:
            coords_in = ((coords - coord_mean) / coord_std).astype(np.float32)
        else:
            coords_in = coords.astype(np.float32)

        xt_t = torch.from_numpy(coords_in).to(device=device, dtype=ftype)

        p_raw = torch.tensor([list(params)], device=device, dtype=ftype)

        if cfg.normalize_io == 1:
            p_unit, p_feat, _ = _build_param_features(p_raw, target_ranges, use_ratio_feats=use_ratio_feats)
        else:
            p_feat = p_raw
            p_unit, _, _ = _build_param_features(p_raw, target_ranges, use_ratio_feats=False)

        p_raw_list = [p_raw[:, j].reshape(-1) for j in range(spec.param_dim)]
        p_unit_list = [p_unit[:, j].reshape(-1) for j in range(spec.param_dim)]

        # polyn also uses unit params in [-1,1], even when normalize_io=0
        polyn_use_raw = False
        Phi_1 = build_param_basis(
            basis_cfg=basis_cfg,
            params_raw=p_raw_list,
            params_unit=p_unit_list,
            Ms=list(Ms),
            polyn_use_raw=polyn_use_raw,
        )

        Phi = Phi_1.repeat(xt_t.shape[0], 1).contiguous()

        if input_mode == "xt":
            feat_in = xt_t
        else:
            p_feat_rep = p_feat.repeat(xt_t.shape[0], 1)
            feat_in = torch.cat([xt_t, p_feat_rep], dim=1).contiguous()

        W = netW(feat_in)
        u_pred = torch.sum(W * Phi, dim=1, keepdim=True)

        u_true_t = torch.from_numpy(u_true.astype(np.float32)).to(device=device, dtype=ftype)

        num = torch.linalg.vector_norm(u_pred - u_true_t)
        den = torch.linalg.vector_norm(u_true_t) + eps_den
        l2re = float((num / den).item())
        l2_list.append(l2re)

        param_str = _format_param_line(param_names, params)
        lines.append(f"{param_str} -> {l2re:.8f}")

    l2_arr = np.asarray(l2_list, dtype=np.float64)
    avg_l2 = float(np.mean(l2_arr)) if l2_arr.size > 0 else float("nan")
    if l2_arr.size >= 2:
        std_l2 = float(np.std(l2_arr, ddof=1))
    elif l2_arr.size == 1:
        std_l2 = 0.0
    else:
        std_l2 = float("nan")

    lines.append(f"avg_L2RE ± std: {avg_l2:.8f} ± {_format_std(std_l2)}")

    txt = "\n".join(lines)
    print("\n" + txt + "\n")

    out_path = os.path.join(folder_path, "final_test.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(txt + "\n")

    print(f"[Final Test] saved: {out_path}")


def train_supervised(spec, cfg: TrainConfig):
    if spec.param_dim not in (1, 2, 3):
        raise ValueError("spec.param_dim must be 1,2, or 3")
    if spec.coord_dim not in (2, 3):
        raise ValueError("spec.coord_dim must be 2 (xt) or 3 (xyt)")
    if len(cfg.Ms) != spec.param_dim:
        raise ValueError(f"len(cfg.Ms) must equal spec.param_dim ({spec.param_dim})")
    if len(cfg.test_params) != spec.param_dim:
        raise ValueError(f"len(cfg.test_params) must equal spec.param_dim ({spec.param_dim})")
    if cfg.normalize_io not in (0, 1):
        raise ValueError("cfg.normalize_io must be 0 or 1")

    set_seed(cfg.seed)

    device = torch.device("cuda:" + cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    ftype = torch.float32

    switch_iter = int(0.8 * cfg.iters)
    use_amp_global = bool(cfg.amp80 and (device.type == "cuda"))
    amp_dtype = torch.float16
    print(
        f"[AMP schedule] amp80={cfg.amp80}, use_amp_first_80%={use_amp_global}, "
        f"amp_dtype={amp_dtype}, switch_iter={switch_iter}/{cfg.iters}"
    )

    param_table = np.asarray(spec.train_param_cases, dtype=np.float32)
    if param_table.ndim != 2:
        raise ValueError("spec.train_param_cases must be 2D array (num_cases, param_dim)")
    if param_table.shape[1] != spec.param_dim:
        raise ValueError(
            f"spec.train_param_cases second dim must be param_dim={spec.param_dim} (got {param_table.shape[1]})"
        )
    num_cases = int(param_table.shape[0])
    if num_cases <= 0:
        raise ValueError("spec.train_param_cases is empty; num_cases must be > 0")

    if cfg.pretrain:
        if cfg.target_param_ranges is None or len(cfg.target_param_ranges) != spec.param_dim:
            raise ValueError("pretrain=True requires target_param_ranges with len=param_dim")
        target_ranges = cfg.target_param_ranges
    else:
        target_ranges = _ranges_from_cases(param_table)

    if cfg.input_mode == "auto":
        max_target = max(r[1] for r in target_ranges)
        input_mode = "xtp" if max_target >= cfg.auto_threshold else "xt"
    else:
        input_mode = cfg.input_mode

    coord_dim = spec.coord_dim
    if input_mode not in {"xt", "xtp"}:
        raise ValueError("input_mode must be 'xt', 'xtp', or 'auto'")

    if input_mode == "xt":
        param_feat_dim = 0
    else:
        if cfg.normalize_io == 1:
            param_feat_dim = spec.param_dim + (
                (spec.param_dim * (spec.param_dim - 1)) // 2 if cfg.use_ratio_feats else 0
            )
        else:
            param_feat_dim = spec.param_dim

    in_dim = coord_dim if input_mode == "xt" else (coord_dim + param_feat_dim)

    print(
        f"Cases: {num_cases}, coord_dim={coord_dim}, param_dim={spec.param_dim}, input_mode={input_mode}, "
        f"use_ratio_feats={cfg.use_ratio_feats}, normalize_io={cfg.normalize_io}, in_dim={in_dim}"
    )

    first_params = tuple(float(x) for x in param_table[0].tolist())
    coords_full, _ = spec.load_case(cfg.data_dir, *first_params)
    if coords_full.shape[1] != coord_dim:
        raise ValueError(f"Loaded coords dim {coords_full.shape[1]} != spec.coord_dim {coord_dim}")

    N_total = coords_full.shape[0]
    Pxt = min(cfg.xt_points, N_total)

    def sample_xt_idx_np():
        return np.random.choice(N_total, size=Pxt, replace=False).astype(np.int64)

    if cfg.normalize_io == 1:
        coord_mean = coords_full.mean(axis=0)
        coord_std = coords_full.std(axis=0)
        coord_std[coord_std < 1e-8] = 1.0
        coords_full_norm = ((coords_full - coord_mean) / coord_std).astype(np.float32)
        print("[normalize_io] ON: standardize coords; net params use unit scaling (and optional ratio feats).")
    else:
        coord_mean = np.zeros((coord_dim,), dtype=np.float32)
        coord_std = np.ones((coord_dim,), dtype=np.float32)
        coords_full_norm = coords_full.astype(np.float32)
        print("[normalize_io] OFF: raw coords; net params are raw (no ratio feats).")

    if cfg.resample_xt:
        idx_xt_fixed = None
        print(f"[coords] resample_xt=ON: xt sampled every {cfg.resample_xt_interval}")
    else:
        idx_xt_fixed = sample_xt_idx_np()
        print("[coords] resample_xt=OFF: fixed subset used only for sampling")

    mins = coords_full_norm.min(axis=0)
    maxs = coords_full_norm.max(axis=0)
    if coord_dim == 2:
        print(f"[coords_in] x in [{mins[0]:.6f}, {maxs[0]:.6f}], t in [{mins[1]:.6f}, {maxs[1]:.6f}]")
    else:
        print(
            f"[coords_in] x in [{mins[0]:.6f}, {maxs[0]:.6f}], "
            f"y in [{mins[1]:.6f}, {maxs[1]:.6f}], "
            f"t in [{mins[2]:.6f}, {maxs[2]:.6f}]"
        )

    print("Loading all solutions...")
    U_all_cpu = np.empty((num_cases, N_total), dtype=np.float32)
    case_params = param_table.astype(np.float32)

    pbar = tqdm(total=num_cases, desc="Loading solutions")
    for cid in range(num_cases):
        params = tuple(float(x) for x in case_params[cid].tolist())
        sol_flat = spec.load_solution_only(cfg.data_dir, *params)
        if sol_flat.shape[0] != N_total:
            raise ValueError(f"Solution length mismatch at case {cid}: {sol_flat.shape[0]} != {N_total}")
        U_all_cpu[cid, :] = sol_flat[:, 0]
        pbar.update(1)
    pbar.close()

    tol = 1e-6
    is_int = np.all(np.abs(case_params - np.round(case_params)) < tol, axis=1)
    int_case_ids = np.where(is_int)[0].astype(np.int64)
    all_case_ids = np.arange(num_cases, dtype=np.int64)
    non_int_case_ids = np.setdiff1d(all_case_ids, int_case_ids, assume_unique=False)
    if len(int_case_ids) == 0:
        int_case_ids = all_case_ids.copy()
        non_int_case_ids = np.empty((0,), dtype=np.int64)
        print("[case pool] integer pool empty => curriculum disabled (using all cases for all stages).")
    else:
        print(f"[case pool] integer-only: {len(int_case_ids)}/{num_cases}, non-integer: {len(non_int_case_ids)}/{num_cases}")

    Ms = list(int(m) for m in cfg.Ms)
    K = compute_K(Ms)
    print(f"[Basis] kind={cfg.basis}, Ms={Ms}, K={K}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder = (
        f"pde-{spec.name}_coord{coord_dim}d_param{spec.param_dim}d"
        f"_basis-{cfg.basis}_in{input_mode}{in_dim}"
        f"_Ms-{'-'.join(map(str,Ms))}"
        f"_loss-{cfg.loss_mode}"
        f"_iter{int(cfg.iters/1000)}k_lr{cfg.lr}"
        f"_points{cfg.xt_points}"
        f"_nio-{int(cfg.normalize_io)}"
        + (f"_ampfp16" if cfg.amp80 else "")
    )
    path_str = os.path.join(base_dir, cfg.save_dir, folder)
    ensure_dir(path_str)
    print("Save dir:", path_str)

    netW = WNet(layer_num=cfg.layers, hidden_size=cfg.hidden_size, out_dim=K, in_features=in_dim, act=cfg.act)
    netW.initialize()
    netW.to(device)

    if cfg.warm_start_path:
        ckpt = torch.load(cfg.warm_start_path, map_location="cpu")
        ckpt_Ms = ckpt.get("Ms", None)
        if ckpt_Ms is not None and tuple(int(x) for x in ckpt_Ms) != tuple(Ms):
            raise ValueError("Warm-start checkpoint Ms mismatch.")
        netW.load_state_dict(ckpt["model_state_dict"], strict=cfg.warm_start_strict)
        print(f"Loaded warm start from {cfg.warm_start_path}")

    coords_full_norm_t = torch.from_numpy(coords_full_norm).to(device=device, dtype=ftype)
    case_params_t = torch.from_numpy(case_params).to(device=device, dtype=ftype)

    params_unit_t, case_feat_unit_t, feat_names_unit = _build_param_features(
        case_params_t, target_ranges, use_ratio_feats=cfg.use_ratio_feats
    )

    if cfg.normalize_io == 1:
        case_feat_t = case_feat_unit_t
        feat_names = feat_names_unit
    else:
        case_feat_t = case_params_t.contiguous()
        feat_names = [f"raw_p{i}" for i in range(spec.param_dim)]

    basis_cfg = BasisConfig(kind=cfg.basis)

    with torch.no_grad():
        params_raw_list = [case_params_t[:, i].contiguous() for i in range(spec.param_dim)]
        params_unit_list = [params_unit_t[:, i].contiguous() for i in range(spec.param_dim)]
        # polyn also uses unit params in [-1,1], even when normalize_io=0
        polyn_use_raw = False
        case_Phi_all = build_param_basis(
            basis_cfg=basis_cfg,
            params_raw=params_raw_list,
            params_unit=params_unit_list,
            Ms=Ms,
            polyn_use_raw=polyn_use_raw,
        )

    if cfg.U_all_on_gpu == 1:
        U_all_t = torch.from_numpy(U_all_cpu).to(device=device, dtype=ftype).contiguous()
        U_all_cpu = None
        print("U_all on GPU:", tuple(U_all_t.shape))
    else:
        U_all_t = torch.from_numpy(U_all_cpu).contiguous()
        print("U_all on CPU:", tuple(U_all_t.shape))

    test_params = tuple(float(x) for x in cfg.test_params)
    test_coords, test_u = spec.load_case(cfg.data_dir, *test_params)
    if test_coords.shape[1] != coord_dim:
        raise ValueError("Test coords dim mismatch.")

    if cfg.normalize_io == 1:
        test_coords_in = ((test_coords - coord_mean) / coord_std).astype(np.float32)
    else:
        test_coords_in = test_coords.astype(np.float32)

    test_u_t = torch.from_numpy(test_u.astype(np.float32)).to(device=device, dtype=ftype)
    test_xt_t = torch.from_numpy(test_coords_in).to(device=device, dtype=ftype)

    if input_mode == "xt":
        test_feat_in_t = test_xt_t
        p_raw = None
        p_unit = None
    else:
        p_raw = torch.tensor([list(test_params)], device=device, dtype=ftype)

        if cfg.normalize_io == 1:
            p_unit, p_feat, _ = _build_param_features(p_raw, target_ranges, use_ratio_feats=cfg.use_ratio_feats)
        else:
            p_feat = p_raw
            p_unit, _, _ = _build_param_features(p_raw, target_ranges, use_ratio_feats=False)

        p_feat_rep = p_feat.repeat(test_xt_t.shape[0], 1)
        test_feat_in_t = torch.cat([test_xt_t, p_feat_rep], dim=1).contiguous()

    with torch.no_grad():
        if input_mode == "xt":
            p_raw = torch.tensor([list(test_params)], device=device, dtype=ftype)
            p_unit, _, _ = _build_param_features(p_raw, target_ranges, use_ratio_feats=False)

        p_raw_list = [p_raw[:, i].reshape(-1) for i in range(spec.param_dim)]
        p_unit_list = [p_unit[:, i].reshape(-1) for i in range(spec.param_dim)]
        polyn_use_raw = False
        test_Phi_1 = build_param_basis(
            basis_cfg=basis_cfg,
            params_raw=p_raw_list,
            params_unit=p_unit_list,
            Ms=Ms,
            polyn_use_raw=polyn_use_raw,
        )
        test_Phi = test_Phi_1.repeat(test_feat_in_t.shape[0], 1).contiguous()

    optimizer = optim.Adam(netW.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    mid_iter = cfg.iters // 2
    n1 = _count_lr_updates_in_range(1, mid_iter, cfg.eval_interval, exclude_zero=True)
    n2 = _count_lr_updates_in_range(mid_iter, cfg.iters, cfg.eval_interval, exclude_zero=True)

    gamma1 = _gamma_for_target_ratio(0.1, n1)
    gamma2 = _gamma_for_target_ratio(1.0 / 10.0, n2)

    print(
        f"[LR schedule] eval_interval={cfg.eval_interval}, iters={cfg.iters}, mid_iter={mid_iter}, "
        f"n_updates_stage1={n1}, n_updates_stage2={n2}"
    )
    print(f"[LR schedule] gamma1={gamma1:.10f}, gamma2={gamma2:.10f} (cfg.lr_gamma1/2 ignored; auto computed)")

    INT_STAGE1 = 0.10
    INT_STAGE2 = 0.20
    INT_MIX = 0.50

    def sample_case_idx_np(B: int, epoch: int) -> np.ndarray:
        B = int(B)
        if B <= 0:
            return np.empty((0,), dtype=np.int64)
        if B >= num_cases:
            return all_case_ids.copy()

        pprog = epoch / float(cfg.iters)

        if pprog < INT_STAGE1:
            pool = int_case_ids
        elif pprog < INT_STAGE2 and len(non_int_case_ids) > 0:
            n_int = min(int(round(B * INT_MIX)), len(int_case_ids))
            idx_int = np.random.choice(int_case_ids, size=n_int, replace=False).astype(np.int64)
            n_rem = B - n_int
            if n_rem <= 0:
                np.random.shuffle(idx_int)
                return idx_int
            idx_non = np.random.choice(non_int_case_ids, size=n_rem, replace=False).astype(np.int64)
            idx = np.concatenate([idx_int, idx_non], axis=0)
            np.random.shuffle(idx)
            return idx
        else:
            pool = all_case_ids

        return np.random.choice(pool, size=B, replace=False).astype(np.int64)

    def build_batch_t(idx_case_np: np.ndarray, idx_xt_np: np.ndarray):
        idx_case_t = torch.from_numpy(idx_case_np).to(device=device, dtype=torch.long)
        idx_xt_t = torch.from_numpy(idx_xt_np).to(device=device, dtype=torch.long)

        coord_part = coords_full_norm_t.index_select(0, idx_xt_t)
        B = idx_case_t.shape[0]
        coord_rep = coord_part.unsqueeze(0).expand(B, -1, -1).reshape(B * Pxt, coord_dim).contiguous()

        phi_t = case_Phi_all.index_select(0, idx_case_t)

        if U_all_t.device.type == "cuda":
            u_bp = U_all_t.index_select(0, idx_case_t).index_select(1, idx_xt_t)
        else:
            u_bp = U_all_t.index_select(0, idx_case_t.cpu()).index_select(1, idx_xt_t.cpu())
            u_bp = u_bp.to(device=device, dtype=ftype)
        u_t = u_bp.reshape(-1, 1).contiguous()

        if input_mode == "xt":
            feat_in = coord_rep
        else:
            pfeat = case_feat_t.index_select(0, idx_case_t)
            pfeat_rep = pfeat.repeat_interleave(Pxt, dim=0)
            feat_in = torch.cat([coord_rep, pfeat_rep], dim=1).contiguous()

        return feat_in, phi_t, u_t, B

    def _one_step_oom_test(B_try: int, idx_xt_np: np.ndarray, amp_enabled_test: bool) -> bool:
        idx_case_np = np.random.choice(num_cases, size=B_try, replace=False).astype(np.int64)
        try:
            netW.train()
            optimizer.zero_grad(set_to_none=True)
            feat_t, phi_t, u_t, Bb = build_batch_t(idx_case_np, idx_xt_np)

            if amp_enabled_test:
                assert device.type == "cuda"
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    W = netW(feat_t)
                    W3 = W.reshape(Bb, Pxt, K)
                    pred2 = (W3 * phi_t[:, None, :]).sum(dim=2)
                    pred = pred2.reshape(-1, 1)
                    if cfg.loss_mode == "mse":
                        loss = torch.mean((pred - u_t) ** 2)
                    else:
                        loss, _, _ = loss_mse_nmae(
                            pred,
                            u_t,
                            eps=cfg.nmae_eps,
                            mix_lambda=cfg.mix_lambda,
                            per_case=cfg.per_case_loss,
                            Bcase=Bb,
                            Pxt=Pxt,
                        )
                scaler.scale(loss).backward()
            else:
                W = netW(feat_t)
                W3 = W.reshape(Bb, Pxt, K)
                pred2 = (W3 * phi_t[:, None, :]).sum(dim=2)
                pred = pred2.reshape(-1, 1)
                if cfg.loss_mode == "mse":
                    loss = torch.mean((pred - u_t) ** 2)
                else:
                    loss, _, _ = loss_mse_nmae(
                        pred,
                        u_t,
                        eps=cfg.nmae_eps,
                        mix_lambda=cfg.mix_lambda,
                        per_case=cfg.per_case_loss,
                        Bcase=Bb,
                        Pxt=Pxt,
                    )
                loss.backward()

            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            return False
        except torch.cuda.OutOfMemoryError:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return True

    def find_max_Bcase(idx_xt_np: np.ndarray, amp_enabled_search: bool, B_start: int) -> int:
        B_start = int(max(1, min(B_start, num_cases)))

        if not _one_step_oom_test(B_start, idx_xt_np, amp_enabled_test=amp_enabled_search):
            return B_start

        B_bad = B_start
        B = B_start
        while True:
            B_new = (4 * B) // 5
            if B_new >= B:
                B_new = B - 1
            B = max(1, B_new)

            if B == 1:
                if _one_step_oom_test(1, idx_xt_np, amp_enabled_test=amp_enabled_search):
                    raise RuntimeError("Cannot fit even Bcase=1. Reduce Pxt/hidden_size/K.")
                return 1

            if not _one_step_oom_test(B, idx_xt_np, amp_enabled_test=amp_enabled_search):
                B_ok = B
                break

        lo, hi = B_ok, B_bad
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if _one_step_oom_test(mid, idx_xt_np, amp_enabled_test=amp_enabled_search):
                hi = mid
            else:
                lo = mid
        return lo

    if cfg.resample_xt:
        idx_xt = sample_xt_idx_np()
        xt_resample_enabled = True
    else:
        idx_xt = idx_xt_fixed
        xt_resample_enabled = False

    amp_for_capacity = bool(use_amp_global and device.type == "cuda")
    if cfg.case_batch > 0:
        B_upper = min(cfg.case_batch, num_cases)
    else:
        B_upper = num_cases

    can_try_full = (B_upper == num_cases)

    if can_try_full:
        if not _one_step_oom_test(num_cases, idx_xt, amp_enabled_test=amp_for_capacity):
            Bcase = num_cases
            case_resample_enabled = False
            idx_case = np.arange(num_cases, dtype=np.int64)
            print(f"[case] full batch fits: Bcase={Bcase} (no sampling)")
        else:
            Bmax_amp = find_max_Bcase(idx_xt, amp_enabled_search=amp_for_capacity, B_start=num_cases)
            Bcase = max(1, Bmax_amp - 10)
            case_resample_enabled = (Bcase < num_cases)
            idx_case = sample_case_idx_np(Bcase, epoch=0)
            print(f"[case] full batch OOM: Bmax={Bmax_amp}, Bcase(train)={Bcase} (safety -10)")
    else:
        if not _one_step_oom_test(B_upper, idx_xt, amp_enabled_test=amp_for_capacity):
            Bcase = B_upper
            case_resample_enabled = True
            idx_case = sample_case_idx_np(Bcase, epoch=0)
            print(f"[case] capped fits: Bcase={Bcase}")
        else:
            Bmax_amp = find_max_Bcase(idx_xt, amp_enabled_search=amp_for_capacity, B_start=B_upper)
            Bcase = max(1, Bmax_amp - 10)
            case_resample_enabled = True
            idx_case = sample_case_idx_np(Bcase, epoch=0)
            print(f"[case] capped OOM: Bmax={Bmax_amp}, Bcase(train)={Bcase} (safety -10)")

    print(
        f"[Train setup] Bcase={Bcase}, P={Pxt}, coord_dim={coord_dim}, param_dim={spec.param_dim}, "
        f"input_mode={input_mode}, K={K}, xt_resample={xt_resample_enabled}, case_resample={case_resample_enabled}"
    )

    np.savez(
        os.path.join(path_str, "normalization_params.npz"),
        coord_mean=coord_mean,
        coord_std=coord_std,
        train_param_cases=case_params.astype(np.float32),
        target_param_ranges=np.array(target_ranges, dtype=np.float32),
        Ms=np.array(Ms, dtype=np.int32),
        input_mode=np.array([input_mode]),
        coord_dim=np.int32(coord_dim),
        param_dim=np.int32(spec.param_dim),
        U_all_on_gpu=np.array([cfg.U_all_on_gpu], dtype=np.int32),
        act=np.array([cfg.act]),
        resample_xt=np.array([int(cfg.resample_xt)], dtype=np.int32),
        resample_xt_interval=np.int32(cfg.resample_xt_interval),
        resample_case_interval=np.int32(cfg.resample_case_interval),
        Bcase=np.int32(Bcase),
        P=np.int32(Pxt),
        amp_first80_fp16=np.array([int(use_amp_global)], dtype=np.int32),
        amp_switch_iter=np.int32(switch_iter),
        basis=np.array([cfg.basis]),
        pde=np.array([spec.name]),
        param_names=np.array(list(spec.param_names)),
        use_ratio_feats=np.array([int(cfg.use_ratio_feats)], dtype=np.int32),
        param_feat_names=np.array(feat_names),
        gamma1=np.array([gamma1], dtype=np.float64),
        gamma2=np.array([gamma2], dtype=np.float64),
        amp80=np.array([int(cfg.amp80)], dtype=np.int32),
        normalize_io=np.array([int(cfg.normalize_io)], dtype=np.int32),
    )

    loss_list, mse_list, nmae_list, l2re_list = [], [], [], []
    switched_to_fp32 = False

    for epoch in tqdm(range(cfg.iters)):
        amp_enabled = bool(use_amp_global and (epoch < switch_iter) and (device.type == "cuda"))

        if cfg.amp80 and (not switched_to_fp32) and (epoch >= switch_iter):
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            if cfg.case_batch > 0:
                B_upper_fp32 = min(cfg.case_batch, num_cases)
            else:
                B_upper_fp32 = min(Bcase + 10, num_cases)

            if not _one_step_oom_test(B_upper_fp32, idx_xt, amp_enabled_test=False):
                Bcase_fp32 = B_upper_fp32
                print(f"\n[FP32 switch] Bcase={Bcase_fp32}")
            else:
                Bmax_fp32 = find_max_Bcase(idx_xt, amp_enabled_search=False, B_start=B_upper_fp32)
                Bcase_fp32 = max(1, Bmax_fp32 - 10)
                print(f"\n[FP32 switch] Bmax_fp32={Bmax_fp32}, Bcase_fp32={Bcase_fp32} (safety -10)")

            if Bcase_fp32 < Bcase:
                Bcase = Bcase_fp32
                idx_case = sample_case_idx_np(Bcase, epoch)

            switched_to_fp32 = True
            print(f"\n[Switch @ iter {epoch}] Entering last 20%: AMP=OFF (FP32)\n")

        if xt_resample_enabled and (epoch % cfg.resample_xt_interval == 0):
            idx_xt = sample_xt_idx_np()

        case_interval_now = cfg.resample_case_interval
        if epoch >= switch_iter:
            case_interval_now = max(1, cfg.resample_case_interval // 2)

        if case_resample_enabled and (epoch % case_interval_now == 0):
            idx_case = sample_case_idx_np(Bcase, epoch)

        max_retries = 50
        attempt = 0

        while True:
            try:
                netW.train()
                optimizer.zero_grad(set_to_none=True)

                feat_t, phi_t, u_t, Bb = build_batch_t(idx_case, idx_xt)

                if amp_enabled:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        W = netW(feat_t)
                        W3 = W.reshape(Bb, Pxt, K)
                        pred2 = (W3 * phi_t[:, None, :]).sum(dim=2)
                        pred = pred2.reshape(-1, 1)

                        if cfg.loss_mode == "mse":
                            loss = torch.mean((pred - u_t) ** 2)
                            mse = loss.detach()
                            nmae = torch.tensor(0.0, device=device)
                        else:
                            loss, mse, nmae = loss_mse_nmae(
                                pred,
                                u_t,
                                eps=cfg.nmae_eps,
                                mix_lambda=cfg.mix_lambda,
                                per_case=cfg.per_case_loss,
                                Bcase=Bb,
                                Pxt=Pxt,
                            )

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    W = netW(feat_t)
                    W3 = W.reshape(Bb, Pxt, K)
                    pred2 = (W3 * phi_t[:, None, :]).sum(dim=2)
                    pred = pred2.reshape(-1, 1)

                    if cfg.loss_mode == "mse":
                        loss = torch.mean((pred - u_t) ** 2)
                        mse = loss.detach()
                        nmae = torch.tensor(0.0, device=device)
                    else:
                        loss, mse, nmae = loss_mse_nmae(
                            pred,
                            u_t,
                            eps=cfg.nmae_eps,
                            mix_lambda=cfg.mix_lambda,
                            per_case=cfg.per_case_loss,
                            Bcase=Bb,
                            Pxt=Pxt,
                        )

                    loss.backward()
                    optimizer.step()

                break

            except torch.cuda.OutOfMemoryError:
                if device.type != "cuda":
                    raise
                attempt += 1
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

                new_Bcase = max(1, int(Bcase * 0.95))
                if new_Bcase == Bcase:
                    raise RuntimeError(
                        "OOM even at Bcase=1. Reduce xt_points/hidden_size/Ms/K or move U_all to CPU."
                    )
                Bcase = new_Bcase
                if Bcase < num_cases:
                    case_resample_enabled = True
                idx_case = sample_case_idx_np(Bcase, epoch)
                print(f"[OOM] iter={epoch}, attempt={attempt}: reduce Bcase to {Bcase} and retry.")
                if attempt >= max_retries:
                    raise RuntimeError(f"Exceeded max_retries={max_retries} at iter={epoch}. Last Bcase={Bcase}.")
                continue

        loss_list.append(float(loss.item()))
        mse_list.append(float(mse.item()))
        nmae_list.append(float(nmae.item()))

        if epoch % cfg.eval_interval == 0:
            netW.eval()
            with torch.no_grad():
                Wt = netW(test_feat_in_t)
                test_pred = torch.sum(Wt * test_Phi, dim=1, keepdim=True)
                num = torch.linalg.vector_norm(test_pred - test_u_t)
                den = torch.linalg.vector_norm(test_u_t) + 1e-12
                l2re = (num / den).item()
                l2re_list.append(float(l2re))

                if epoch > 0:
                    gamma = gamma1 if epoch < mid_iter else gamma2
                    for pg in optimizer.param_groups:
                        pg["lr"] *= gamma

                lr_now = optimizer.param_groups[0]["lr"]
                phase_tag = "AMP" if amp_enabled else "FP32"
                print(
                    f"\nIter {epoch}: {phase_tag}, "
                    f"Loss={loss.item():.6e}, MSE={mse.item():.6e}, "
                    f"L2re(test)={l2re:.6f}, LR={lr_now:.2e}"
                )

    checkpoint = {
        "epoch": cfg.iters,
        "model_state_dict": netW.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "coord_mean": coord_mean,
        "coord_std": coord_std,
        "Ms": Ms,
        "train_param_cases": case_params.astype(np.float32),
        "target_param_ranges": [list(x) for x in target_ranges],
        "input_mode": input_mode,
        "coord_dim": coord_dim,
        "param_dim": spec.param_dim,
        "in_dim": in_dim,
        "K": K,
        "U_all_on_gpu": int(cfg.U_all_on_gpu),
        "act": cfg.act,
        "N_total": int(N_total),
        "num_cases": int(num_cases),
        "Bcase": int(Bcase),
        "P": int(Pxt),
        "resample_xt": bool(cfg.resample_xt),
        "resample_case": bool(case_resample_enabled),
        "amp_first80_fp16": bool(use_amp_global),
        "amp_switch_iter": int(switch_iter),
        "basis": cfg.basis,
        "pde": spec.name,
        "param_names": list(spec.param_names),
        "use_ratio_feats": bool(cfg.use_ratio_feats),
        "param_feat_names": feat_names,
        "gamma1": float(gamma1),
        "gamma2": float(gamma2),
        "lr_target_mid_ratio": 0.1,
        "lr_target_end_ratio": 1.0 / 200.0,
        "amp80": bool(cfg.amp80),
        "normalize_io": int(cfg.normalize_io),
    }
    torch.save(checkpoint, os.path.join(path_str, "checkpoint.pth"))

    pd.DataFrame({"loss": loss_list, "mse": mse_list, "nmae": nmae_list}).to_csv(
        os.path.join(path_str, "loss.csv"), index=False
    )
    pd.DataFrame({"l2re": l2re_list}).to_csv(os.path.join(path_str, "l2re.csv"), index=False)

    print("Saved to:", path_str)

    final_test_after_training(
        spec=spec,
        cfg=cfg,
        folder_path=path_str,
        folder_name=os.path.basename(path_str),
        device=device,
        coord_mean=coord_mean,
        coord_std=coord_std,
        target_ranges=target_ranges,
        Ms=Ms,
        input_mode=input_mode,
        use_ratio_feats=cfg.use_ratio_feats,
        act=cfg.act,
    )