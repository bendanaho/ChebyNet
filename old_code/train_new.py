# - 模型目标：学习参数化映射 u(x,t;β,ν,ρ)。网络输出每个样本点对应的 Chebyshev 展开权重 W，
#   再与预计算的三维 Chebyshev 基 Φ(β,ν,ρ) 组合得到 u 的预测值。
#
# - 输入特征：
#   1) xt 模式：归一化后的 (x,t)（2 维）。
#   2) xtp 模式：归一化后的 (x,t) + 6 维参数特征
#      [βu, νu, ρu, log(1+β/ν), log(1+ρ/ν), log(1+ρ/β)]（共 8 维）。
#
# - 输出形式：输出 Chebyshev 展开系数权重向量 W ∈ R^K，
#   其中 K = (M_beta+1)(M_nu+1)(M_rho+1)。
#
# - 基函数表示：对每个 case 预计算 Φ(β,ν,ρ)（三维各向异性 Chebyshev 基），
#   预测为 u_hat(x,t) = sum_{k=1..K} W_k(x,t,·) * Φ_k(β,ν,ρ)。
#
# - loss 计算：
#   * mse：MSE = E[(u_hat - u)^2]
#   * mix：L = λ*MSE + (1-λ)*NMAE，其中 NMAE = E[ |u_hat-u| / (|u|+eps) ]
#   * 支持 per-case：先在每个 case 内对 Pxt 个点求均值，再对 Bcase 个 case 求平均。
#
# - 采样策略：每次迭代从全网格中抽取 Pxt 个 (x,t) 点；case 维度按 Bcase 采样（可全量或子集），
#   并采用课程采样：整数参数 case 优先 → 混合 → 全量随机。
#
# - 重采样机制：支持按间隔重采样 xt 索引（resample_xt，默认开启）与重采样 case（resample_case_interval）。
#
# - 训练与加速：前 80% 迭代使用 AMP(fp16) 加速，后 20% 切换 FP32 提升稳定性；
#   包含 OOM 自适应逻辑（自动缩小 Bcase 并重试）。
#
# - 优化器与学习率：Adam；学习率在评估点按分段指数衰减（前半程 lr_gamma1，后半程 lr_gamma2）。
#
# - 评估指标：在指定测试参数 (β,ν,ρ) 上定期计算 L2re = ||u_hat-u||_2 / (||u||_2 + tiny)。
#
# - 保存内容：保存归一化参数、训练日志（loss/mse/nmae、l2re）以及包含模型权重与关键超参的 checkpoint。
# -----------------------------------------------------------------------------------------------------------------------
# - 推荐的GPU型号：NVIDIA RTX 5090
# - 推荐的训练设置（推荐cheb基底）（可根据资源调整）：
# [1,5]		python train_cheb.py --basis cheb --layers 5 --hidden_size 128 --M_beta 5 --M_nu 5 --M_rho 5 --range_beta 1 5 --range_nu 1 5 --range_rho 1 5 --interval 0.5 --input_mode xt --lr_gamma1 0.99 --lr_gamma2 0.987 --loss_mode mse --xt_points 1000 --resample_case_interval 1000 --resample_xt_interval 5000 --U_all_on_gpu 1
# [1,10]	python train_cheb.py --basis cheb --layers 6 --hidden_size 192 --M_beta 6 --M_nu 6 --M_rho 6 --range_beta 1 10 --range_nu 1 10 --range_rho 1 10 --interval 0.5 --input_mode xtp --lr_gamma1 0.99 --lr_gamma2 0.987 --loss_mode mse --xt_points 1000 --case_batch 2000 --resample_case_interval 1000 --resample_xt_interval 5000 --U_all_on_gpu 1
# [1,20]	python train_cheb.py --basis cheb --layers 7 --hidden_size 256 --M_beta 7 --M_nu 7 --M_rho 7 --range_beta 1 20 --range_nu 1 20 --range_rho 1 20 --interval 0.5 --input_mode xtp --lr_gamma1 0.99 --lr_gamma2 0.99 --loss_mode mix --mix_lambda 0.95 --xt_points 1000 --resample_case_interval 1000 --resample_xt_interval 5000 --U_all_on_gpu 1
# polyn版本（同样的 M 和其他设置）：
# [1,5]		python train_cheb.py --basis polyn --clamp_size 20 --layers 5 --hidden_size 128 --M_beta 5 --M_nu 5 --M_rho 5 --range_beta 1 5 --range_nu 1 5 --range_rho 1 5 --interval 0.5 --input_mode xt --lr_gamma1 0.99 --lr_gamma2 0.987 --loss_mode mse --xt_points 1000 --resample_case_interval 1000 --resample_xt_interval 5000 --U_all_on_gpu 1
# [1,10]	python train_cheb.py --basis polyn --clamp_size 50 --layers 6 --hidden_size 192 --M_beta 6 --M_nu 6 --M_rho 6 --range_beta 1 10 --range_nu 1 10 --range_rho 1 10 --interval 0.5 --input_mode xtp --lr_gamma1 0.99 --lr_gamma2 0.987 --loss_mode mix --mix_lambda 0.95 --xt_points 1000 --case_batch 2000 --resample_case_interval 1000 --resample_xt_interval 5000 --U_all_on_gpu 1
# [1,20]	python train_cheb.py --basis polyn --clamp_size 100 --layers 7 --hidden_size 256 --M_beta 7 --M_nu 7 --M_rho 7 --range_beta 1 20 --range_nu 1 20 --range_rho 1 20 --interval 0.5 --input_mode xtp --lr_gamma1 0.99 --lr_gamma2 0.99 --loss_mode mix --mix_lambda 0.95 --xt_points 1000 --resample_case_interval 1000 --resample_xt_interval 5000 --U_all_on_gpu 1

import os
import math
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# ----------------- Model -----------------
class WNet(nn.Module):
    def __init__(
        self,
        layer_num: int,
        hidden_size: int,
        out_dim: int,
        in_features: int,
        act: str = "gelu",
    ):
        super().__init__()
        act = act.lower().strip()
        if act not in {"tanh", "gelu", "gelu_tanh", "silu"}:
            raise ValueError("act must be one of: tanh, gelu, gelu_tanh, silu")

        self.act_name = act  # for logging / saving

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

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# ----------------- Reproducibility -----------------
def set_seed(myseed: int):
    random.seed(myseed)
    os.environ["PYTHONHASHSEED"] = str(myseed)
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    torch.cuda.manual_seed(myseed)
    torch.cuda.manual_seed_all(myseed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ----------------- Chebyshev basis -----------------
def scale_to_unit_torch(x: torch.Tensor, a: float, b: float):
    return 2.0 * (x - a) / (b - a) - 1.0


def scale_to_unit_np(x: np.ndarray, a: float, b: float):
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


def chebyshev_T_np(x: np.ndarray, M: int) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    N = x.shape[0]
    T = np.empty((N, M + 1), dtype=np.float64)
    T[:, 0] = 1.0
    if M >= 1:
        T[:, 1] = x
    for n in range(2, M + 1):
        T[:, n] = 2.0 * x * T[:, n - 1] - T[:, n - 2]
    return T


def chebyshev_basis_3d_aniso(beta_u, nu_u, rho_u, M_beta, M_nu, M_rho):
    Tb = chebyshev_T(beta_u, M_beta)
    Tn = chebyshev_T(nu_u, M_nu)
    Tr = chebyshev_T(rho_u, M_rho)
    Phi = torch.einsum("bi,bj,bk->bijk", Tb, Tn, Tr).reshape(beta_u.shape[0], -1)
    return Phi

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
    Tn = torch.stack([nu   ** j for j in range(M_nu + 1)],   dim=1)  # (B, M_nu+1)
    Tr = torch.stack([rho  ** k for k in range(M_rho + 1)],  dim=1)  # (B, M_rho+1)

    Phi = torch.einsum("bi,bj,bk->bijk", Tb, Tn, Tr).reshape(beta.shape[0], -1)
    return Phi


# ----------------- Data helpers -----------------
def case_filename(data_dir: str, beta: float, nu: float, rho: float) -> str:
    return f"{data_dir}/cdr_gt_beta_{beta:.2f}_nu_{nu:.2f}_rho_{rho:.2f}.npz"


def load_case(data_dir: str, beta: float, nu: float, rho: float):
    data = np.load(case_filename(data_dir, beta, nu, rho))
    X = data["x"]
    T = data["t"]
    sol = data["solution"]
    X_grid, T_grid = np.meshgrid(X, T, indexing="ij")
    coords = np.concatenate((X_grid.reshape(-1, 1), T_grid.reshape(-1, 1)), axis=1).astype(np.float32)
    sol_flat = sol.reshape(-1, 1).astype(np.float32)
    return coords, sol_flat


def load_solution_only(data_dir: str, beta: float, nu: float, rho: float):
    data = np.load(case_filename(data_dir, beta, nu, rho))
    sol = data["solution"].astype(np.float32)
    return sol.reshape(-1, 1)


def make_grid(a: float, b: float, step: float, decimals: int = 2):
    if step <= 0:
        raise ValueError("step must be > 0")
    n = int(round((b - a) / step))
    vals = [round(a + i * step, decimals) for i in range(n + 1)]
    vals = [v for v in vals if v <= b + 1e-12]
    vals = sorted(list(dict.fromkeys(vals)))
    return np.array(vals, dtype=np.float32)


# ----------------- Loss -----------------
def loss_mse_nmae(pred, truth, eps=1e-3, mix_lambda=0.7, per_case=False, Bcase=None, Pxt=None):
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


# ----------------- Main -----------------
def main():
    parser = argparse.ArgumentParser()
    # basis
    parser.add_argument("--basis",type=str,default="cheb",choices=["cheb", "polyn"])
    parser.add_argument("--clamp_size",type=float,default=20.0,help="Only used when --basis polyn.")

    # network
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--hidden_size", type=int, default=192)
    parser.add_argument(
        "--act",
        type=str,
        default="gelu",
        choices=["tanh", "gelu", "gelu_tanh", "silu"],
        help="Activation function used in WNet hidden layers.",
    )

    # M (must be specified by user)
    parser.add_argument("--M_beta", type=int, default=6, help="Chebyshev order for beta (>=0).")
    parser.add_argument("--M_nu", type=int, default=6, help="Chebyshev order for nu (>=0).")
    parser.add_argument("--M_rho", type=int, default=6, help="Chebyshev order for rho (>=0).")

    # ranges + step
    parser.add_argument("--range_beta", type=float, nargs=2, default=[1.0, 10.0])
    parser.add_argument("--range_nu", type=float, nargs=2, default=[1.0, 10.0])
    parser.add_argument("--range_rho", type=float, nargs=2, default=[1.0, 10.0])
    parser.add_argument("--interval", type=float, default=0.5)

    # pretrain
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--target_range_beta", type=float, nargs=2, default=None)
    parser.add_argument("--target_range_nu", type=float, nargs=2, default=None)
    parser.add_argument("--target_range_rho", type=float, nargs=2, default=None)

    # input
    parser.add_argument("--input_mode", type=str, default="auto", choices=["auto", "xt", "xtp"])
    parser.add_argument("--auto_threshold", type=float, default=10.0)

    # training
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--iter", type=int, default=200000)
    parser.add_argument("--lr_gamma1", type=float, default=0.99)
    parser.add_argument("--lr_gamma2", type=float, default=0.987)
    parser.add_argument("--eval_interval", type=int, default=500)

    # loss
    parser.add_argument("--loss_mode", type=str, default="mse", choices=["mse", "mix"])
    parser.add_argument("--mix_lambda", type=float, default=0.95)
    parser.add_argument("--nmae_eps", type=float, default=1e-3)
    parser.add_argument("--per_case_loss", action="store_true")

    # sampling
    parser.add_argument("--xt_points", type=int, default=1000)

    # case batching / resample
    parser.add_argument("--case_batch", type=int, default=0, help="If >0, cap the number of cases per iter (auto-shrink on OOM); if 0, auto-fit up to all cases.")
    parser.add_argument("--resample_case_interval", type=int, default=1000)

    # xt resample (default ON now)
    parser.add_argument(
        "--resample_xt",
        action="store_true",
        default=True,
        help="Resample xt indices during training (default: ON).",
    )
    parser.add_argument("--resample_xt_interval", type=int, default=5000)

    # test
    parser.add_argument("--test_beta", type=float, default=3.25)
    parser.add_argument("--test_nu", type=float, default=3.25)
    parser.add_argument("--test_rho", type=float, default=3.25)

    # io
    parser.add_argument("--data_dir", type=str, default="CDR_ground_truth_precise")
    parser.add_argument("--save_dir", type=str, default="CDR_using_new-config")

    # warm start
    parser.add_argument("--warm_start_path", type=str, default="")
    parser.add_argument("--warm_start_strict", action="store_true")

    # speed switch
    parser.add_argument("--U_all_on_gpu", type=int, default=1, help="1: keep U_all on GPU, 0: keep on CPU")

    args = parser.parse_args()

    # Validate M
    if args.M_beta < 0 or args.M_nu < 0 or args.M_rho < 0:
        raise ValueError("Auto-M is removed. Please specify --M_beta/--M_nu/--M_rho (all must be >= 0).")

    set_seed(args.seed)

    device = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    ftype = torch.float32

    # AMP schedule: first 80% AMP fp16, last 20% FP32
    switch_iter = int(0.8 * args.iter)
    use_amp_global = (device.type == "cuda")

    # [MOD] bf16 -> fp16
    amp_dtype = torch.float16
    print(f"[AMP schedule] use_amp_first_80%={use_amp_global}, amp_dtype={amp_dtype}, switch_iter={switch_iter}/{args.iter}")

    # target ranges
    if args.pretrain:
        if args.target_range_beta is None or args.target_range_nu is None or args.target_range_rho is None:
            raise ValueError("pretrain=True requires --target_range_beta/nu/rho")
        target_range_beta = args.target_range_beta
        target_range_nu = args.target_range_nu
        target_range_rho = args.target_range_rho
    else:
        target_range_beta = args.range_beta
        target_range_nu = args.range_nu
        target_range_rho = args.range_rho

    # input mode
    if args.input_mode == "auto":
        max_target = max(target_range_beta[1], target_range_nu[1], target_range_rho[1])
        input_mode = "xtp" if max_target >= args.auto_threshold else "xt"
    else:
        input_mode = args.input_mode
    in_dim = 2 if input_mode == "xt" else 8

    # grids
    beta_range = make_grid(args.range_beta[0], args.range_beta[1], args.interval, decimals=2)
    nu_range = make_grid(args.range_nu[0], args.range_nu[1], args.interval, decimals=2)
    rho_range = make_grid(args.range_rho[0], args.range_rho[1], args.interval, decimals=2)
    Nb, Nn, Nr = len(beta_range), len(nu_range), len(rho_range)
    num_cases = Nb * Nn * Nr
    print(f"Cases: {num_cases} (Nb={Nb}, Nn={Nn}, Nr={Nr}), input_mode={input_mode}, step={args.interval}")

    # coords (raw)
    coords_full, _ = load_case(args.data_dir, float(beta_range[0]), float(nu_range[0]), float(rho_range[0]))
    N_total = coords_full.shape[0]
    Pxt = min(args.xt_points, N_total)

    # sampling helper (needs N_total/Pxt)
    def sample_xt_idx_np():
        return np.random.choice(N_total, size=Pxt, replace=False).astype(np.int64)

    # xt normalization logic (resample_xt default ON)
    if args.resample_xt:
        xt_mean = coords_full.mean(axis=0)
        xt_std = coords_full.std(axis=0)
        xt_std[xt_std < 1e-8] = 1.0
        idx_xt_fixed = None
        print(f"[xt] resample_xt=ON: xt_mean/std computed on full grid, xt will be resampled every {args.resample_xt_interval} iters.")
    else:
        idx_xt_fixed = sample_xt_idx_np()
        xt_sub = coords_full[idx_xt_fixed]  # (Pxt,2)
        xt_mean = xt_sub.mean(axis=0)
        xt_std = xt_sub.std(axis=0)
        xt_std[xt_std < 1e-8] = 1.0
        print("[xt] resample_xt=OFF: xt fixed subset used; xt_mean/std computed on sampled xt subset.")

    coords_full_norm = ((coords_full - xt_mean) / xt_std).astype(np.float32)

    # Load solutions, build U_all and case_params
    print("Loading all solutions...")
    U_all_cpu = np.empty((num_cases, N_total), dtype=np.float32)
    case_params = np.empty((num_cases, 3), dtype=np.float32)

    case_id = 0
    pbar = tqdm(total=num_cases, desc="Loading solutions")
    for beta in beta_range:
        for nu in nu_range:
            for rho in rho_range:
                b, n, r = float(beta), float(nu), float(rho)
                sol_flat = load_solution_only(args.data_dir, b, n, r)
                U_all_cpu[case_id, :] = sol_flat[:, 0]
                case_params[case_id, :] = np.array([b, n, r], dtype=np.float32)
                case_id += 1
                pbar.update(1)
    pbar.close()

    # ----------------- build integer-only case pool -----------------
    # integer pool definition: beta, nu, rho are all integers
    # use tolerance to avoid float issues (although you constructed them by rounding)
    tol = 1e-6
    is_int = np.all(np.abs(case_params - np.round(case_params)) < tol, axis=1)  # (num_cases,)
    int_case_ids = np.where(is_int)[0].astype(np.int64)

    if len(int_case_ids) == 0:
        raise RuntimeError("Integer case pool is empty. Check ranges/interval or integer definition.")

    all_case_ids = np.arange(num_cases, dtype=np.int64)
    non_int_case_ids = np.setdiff1d(all_case_ids, int_case_ids, assume_unique=False)
    print(f"[case pool] non-integer cases: {len(non_int_case_ids)} / {num_cases} "
          f"({len(non_int_case_ids)/num_cases*100:.2f}%)")

    print(f"[case pool] integer-only cases: {len(int_case_ids)} / {num_cases} "
          f"({len(int_case_ids)/num_cases*100:.2f}%)")

    # Manual M
    M_beta, M_nu, M_rho = args.M_beta, args.M_nu, args.M_rho
    print(f"[Manual M] M_beta={M_beta}, M_nu={M_nu}, M_rho={M_rho}")

    K = (M_beta + 1) * (M_nu + 1) * (M_rho + 1)

    # --- save dir ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    rb0, rb1 = args.range_beta
    rn0, rn1 = args.range_nu
    rr0, rr1 = args.range_rho

    folder = (
        f"new-config_basis-{args.basis}_in{input_mode}{in_dim}"
        f"_layers{args.layers}_hs{args.hidden_size}"
        f"_act-{args.act}"
        f"_Mb{M_beta}_Mn{M_nu}_Mr{M_rho}"
        f"_loss-{args.loss_mode}"
        f"_iter{int(args.iter/1000)}k_lr{args.lr}_wd{args.weight_decay}"
        f"_range{rb0:g}-{rb1:g}"
        f"_step{args.interval}"
        f"_points{args.xt_points}"
        f"_xtRs-{int(args.resample_xt)}"
        f"_rscase{args.resample_case_interval}_rsxt{args.resample_xt_interval}"
        f"_ampfp16"
    )

    path_str = os.path.join(base_dir, args.save_dir, folder)
    os.makedirs(path_str, exist_ok=True)
    print("Save dir:", path_str)

    # model
    netW = WNet(layer_num=args.layers, hidden_size=args.hidden_size, out_dim=K, in_features=in_dim, act=args.act)
    netW.initialize()
    netW.to(device)

    # warm start
    if args.warm_start_path:
        ckpt = torch.load(args.warm_start_path, map_location="cpu")
        if (int(ckpt.get("M_beta", -999)), int(ckpt.get("M_nu", -999)), int(ckpt.get("M_rho", -999))) != (
            M_beta,
            M_nu,
            M_rho,
        ):
            raise ValueError("Warm-start checkpoint M mismatch.")
        netW.load_state_dict(ckpt["model_state_dict"], strict=args.warm_start_strict)
        print(f"Loaded warm start from {args.warm_start_path}")

    # tensors on device
    coords_full_norm_t = torch.from_numpy(coords_full_norm).to(device=device, dtype=ftype)
    case_params_t = torch.from_numpy(case_params).to(device=device, dtype=ftype)

    # precompute case features for xtp: (C,6)
    eps = 1e-12
    beta = case_params_t[:, 0:1]
    nu   = case_params_t[:, 1:2]
    rho  = case_params_t[:, 2:3]

    beta_u = scale_to_unit_torch(beta.squeeze(1), target_range_beta[0], target_range_beta[1]).unsqueeze(1)
    nu_u   = scale_to_unit_torch(nu.squeeze(1),   target_range_nu[0],   target_range_nu[1]).unsqueeze(1)
    rho_u  = scale_to_unit_torch(rho.squeeze(1),  target_range_rho[0],  target_range_rho[1]).unsqueeze(1)

    # --- NEW: physics-motivated combos (replace nd_beta/nd_nu/nd_rho) ---
    f1 = torch.log1p(beta / (nu + eps))    # log(1 + beta/nu)
    f2 = torch.log1p(rho  / (nu + eps))    # log(1 + rho/nu)
    f3 = torch.log1p(rho  / (beta + eps))  # log(1 + rho/beta)

    case_feat6_t = torch.cat([beta_u, nu_u, rho_u, f1, f2, f3], dim=1).contiguous()

    # precompute case Phi: (num_cases, K)
    with torch.no_grad():
        if args.basis == "cheb":
            case_Phi_all = chebyshev_basis_3d_aniso(
                beta_u.squeeze(1),
                nu_u.squeeze(1),
                rho_u.squeeze(1),
                M_beta,
                M_nu,
                M_rho,
            ).contiguous()
        else:  # polyn
            case_Phi_all = polynomial_basis_3d(
                beta.squeeze(1),
                nu.squeeze(1),
                rho.squeeze(1),
                M_beta,
                M_nu,
                M_rho,
            ).contiguous()
            case_Phi_all = torch.clamp(case_Phi_all, min=-args.clamp_size, max=args.clamp_size)

    # U_all
    if args.U_all_on_gpu == 1:
        U_all_t = torch.from_numpy(U_all_cpu).to(device=device, dtype=ftype).contiguous()
        U_all_cpu = None
        print("U_all on GPU:", tuple(U_all_t.shape))
    else:
        U_all_t = torch.from_numpy(U_all_cpu).contiguous()
        print("U_all on CPU:", tuple(U_all_t.shape))

    # test tensors
    test_coords, test_u = load_case(args.data_dir, args.test_beta, args.test_nu, args.test_rho)
    test_xt_norm = ((test_coords - xt_mean) / xt_std).astype(np.float32)
    test_u_t = torch.from_numpy(test_u).to(device=device, dtype=ftype)

    if input_mode == "xt":
        test_feat_t = torch.from_numpy(test_xt_norm).to(device=device, dtype=ftype)
    else:
        p = torch.tensor([[args.test_beta, args.test_nu, args.test_rho]], device=device, dtype=ftype)
        tb = scale_to_unit_torch(p[:, 0], target_range_beta[0], target_range_beta[1]).unsqueeze(1)
        tn = scale_to_unit_torch(p[:, 1], target_range_nu[0], target_range_nu[1]).unsqueeze(1)
        tr = scale_to_unit_torch(p[:, 2], target_range_rho[0], target_range_rho[1]).unsqueeze(1)

        eps = 1e-12
        beta = p[:, 0:1]
        nu   = p[:, 1:2]
        rho  = p[:, 2:3]

        f1 = torch.log1p(beta / (nu + eps))
        f2 = torch.log1p(rho  / (nu + eps))
        f3 = torch.log1p(rho  / (beta + eps))

        p6 = torch.cat([tb, tn, tr, f1, f2, f3], dim=1)

        xt_t = torch.from_numpy(test_xt_norm).to(device=device, dtype=ftype)
        p6_rep = p6.repeat(xt_t.shape[0], 1)
        test_feat_t = torch.cat([xt_t, p6_rep], dim=1).contiguous()

    # test Phi
    p = torch.tensor([[args.test_beta, args.test_nu, args.test_rho]], device=device, dtype=ftype)
    with torch.no_grad():
        if args.basis == "cheb":
            tb = scale_to_unit_torch(p[:, 0], target_range_beta[0], target_range_beta[1])
            tn = scale_to_unit_torch(p[:, 1], target_range_nu[0], target_range_nu[1])
            tr = scale_to_unit_torch(p[:, 2], target_range_rho[0], target_range_rho[1])
            test_Phi_1 = chebyshev_basis_3d_aniso(tb, tn, tr, M_beta, M_nu, M_rho)  # (1,K)
        else:  # polyn
            test_Phi_1 = polynomial_basis_3d(
                p[:, 0], p[:, 1], p[:, 2],
                M_beta, M_nu, M_rho
            )  # (1,K)
            test_Phi_1 = torch.clamp(test_Phi_1, min=-args.clamp_size, max=args.clamp_size)
        test_Phi = test_Phi_1.repeat(test_feat_t.shape[0], 1).contiguous()

    # optimizer
    optimizer = optim.Adam(netW.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # [MOD] GradScaler API + enable only when CUDA
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp_global and device.type == "cuda"))

    # ----------------- sampling helpers -----------------
    # curriculum schedule (hard-coded)
    INT_STAGE1 = 0.10   # 0% - 10%: prefer integer pool
    INT_STAGE2 = 0.20   # 10% - 20%: mix
    INT_MIX = 0.50      # in 10%-20%: fraction from integer pool

    def sample_case_idx_np(B: int, epoch: int) -> np.ndarray:
        """
        Case sampling with curriculum, WITHOUT duplicates whenever possible.

        Rules:
        - If B >= num_cases: return all cases [0..num_cases-1] (no duplicates, includes non-integer).
        - Else:
            * p < 0.10: take as many as possible from integer pool (no replacement),
                        then fill remaining from non-integer pool (no replacement).
            * 0.10 <= p < 0.20: take ~50% from integer pool (no replacement),
                                rest from non-integer pool first, then if still short, from remaining full pool.
            * p >= 0.20: sample from full pool (no replacement).
        """
        B = int(B)
        if B <= 0:
            return np.empty((0,), dtype=np.int64)

        # If batch covers (or exceeds) all cases, just use full pool once.
        if B >= num_cases:
            return all_case_ids.copy()  # length = num_cases, no duplicates, includes non-integer

        p = epoch / float(args.iter)

        if p < INT_STAGE1:
            # Stage1: integer-first, then fill with non-integers (all without replacement)
            n_int = min(B, int_case_ids.shape[0])
            idx_int = np.random.choice(int_case_ids, size=n_int, replace=False).astype(np.int64)

            n_rem = B - n_int
            if n_rem > 0:
                # fill from non-integer pool without replacement
                if n_rem > non_int_case_ids.shape[0]:
                    # This shouldn't happen if B < num_cases, but keep safe:
                    # fall back to sampling from all without replacement
                    idx_all = np.random.choice(all_case_ids, size=B, replace=False).astype(np.int64)
                    return idx_all
                idx_fill = np.random.choice(non_int_case_ids, size=n_rem, replace=False).astype(np.int64)
                idx = np.concatenate([idx_int, idx_fill], axis=0)
            else:
                idx = idx_int

            np.random.shuffle(idx)
            return idx

        elif p < INT_STAGE2:
            # Stage2: mix, but still avoid duplicates and avoid replacement
            n_int_target = int(round(B * INT_MIX))
            n_int = min(n_int_target, int_case_ids.shape[0])
            idx_int = np.random.choice(int_case_ids, size=n_int, replace=False).astype(np.int64)

            n_rem = B - n_int
            if n_rem <= 0:
                np.random.shuffle(idx_int)
                return idx_int

            # Prefer sampling remainder from non-integer pool (so we truly mix)
            if n_rem <= non_int_case_ids.shape[0]:
                idx_non = np.random.choice(non_int_case_ids, size=n_rem, replace=False).astype(np.int64)
                idx = np.concatenate([idx_int, idx_non], axis=0)
                np.random.shuffle(idx)
                return idx

            # If non-integer pool is smaller than required (rare, happens if integer pool is huge),
            # take all non-integers, and fill the rest from remaining integers not already picked.
            idx_non = non_int_case_ids.copy()
            still = n_rem - idx_non.shape[0]

            # remaining integers not in idx_int
            int_remaining = np.setdiff1d(int_case_ids, idx_int, assume_unique=False)
            if still > int_remaining.shape[0]:
                # Safety fallback: sample from all without replacement
                idx_all = np.random.choice(all_case_ids, size=B, replace=False).astype(np.int64)
                return idx_all

            idx_more = np.random.choice(int_remaining, size=still, replace=False).astype(np.int64)
            idx = np.concatenate([idx_int, idx_non, idx_more], axis=0)
            np.random.shuffle(idx)
            return idx

        else:
            # Stage3: full pool random (no replacement)
            return np.random.choice(all_case_ids, size=B, replace=False).astype(np.int64)

    def build_batch_t(idx_case_np, idx_xt_np):
        idx_case_t = torch.from_numpy(idx_case_np).to(device=device, dtype=torch.long)
        idx_xt_t = torch.from_numpy(idx_xt_np).to(device=device, dtype=torch.long)

        # xt: (Pxt,2) -> (B*Pxt,2)
        xt_part = coords_full_norm_t.index_select(0, idx_xt_t)  # (Pxt,2)
        B = idx_case_t.shape[0]
        xt_rep = xt_part.unsqueeze(0).expand(B, -1, -1).reshape(B * Pxt, 2).contiguous()

        # Phi: (B,K)
        phi_t = case_Phi_all.index_select(0, idx_case_t)

        # u: (B,Pxt)
        if U_all_t.device.type == "cuda":
            u_bp = U_all_t.index_select(0, idx_case_t).index_select(1, idx_xt_t)  # (B,Pxt)
        else:
            u_bp = U_all_t.index_select(0, idx_case_t.cpu()).index_select(1, idx_xt_t.cpu())
            u_bp = u_bp.to(device=device, dtype=ftype)
        u_t = u_bp.reshape(-1, 1).contiguous()

        # features
        if input_mode == "xt":
            feat_t = xt_rep
        else:
            p6 = case_feat6_t.index_select(0, idx_case_t)  # (B,6)
            p6_rep = p6.repeat_interleave(Pxt, dim=0)  # (B*Pxt,6)
            feat_t = torch.cat([xt_rep, p6_rep], dim=1).contiguous()  # (B*Pxt,8)

        return feat_t, phi_t, u_t, B

    def _one_step_oom_test(B_try: int, idx_xt_np: np.ndarray, amp_enabled_test: bool) -> bool:
        """Return True if OOM occurs for this B_try (single forward/backward). No optimizer step."""
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
                    if args.loss_mode == "mse":
                        loss = torch.mean((pred - u_t) ** 2)
                    else:
                        loss, _, _ = loss_mse_nmae(
                            pred, u_t, eps=args.nmae_eps, mix_lambda=args.mix_lambda,
                            per_case=args.per_case_loss, Bcase=Bb, Pxt=Pxt
                        )
                scaler.scale(loss).backward()
            else:
                W = netW(feat_t)
                W3 = W.reshape(Bb, Pxt, K)
                pred2 = (W3 * phi_t[:, None, :]).sum(dim=2)
                pred = pred2.reshape(-1, 1)
                if args.loss_mode == "mse":
                    loss = torch.mean((pred - u_t) ** 2)
                else:
                    loss, _, _ = loss_mse_nmae(
                        pred, u_t, eps=args.nmae_eps, mix_lambda=args.mix_lambda,
                        per_case=args.per_case_loss, Bcase=Bb, Pxt=Pxt
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
        """
        Find maximum Bcase <= B_start that fits memory.
        Phase1: shrink to find a feasible point.
        Phase2: binary search to reach the true max (avoid wasting memory).
        """
        B_start = int(max(1, min(B_start, num_cases)))

        # If B_start fits, done
        if not _one_step_oom_test(B_start, idx_xt_np, amp_enabled_test=amp_enabled_search):
            return B_start

        # Phase 1: shrink until fits
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

        # Phase 2: binary search (B_ok fits, B_bad OOM)
        lo, hi = B_ok, B_bad
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if _one_step_oom_test(mid, idx_xt_np, amp_enabled_test=amp_enabled_search):
                hi = mid
            else:
                lo = mid
        return lo

    # ----------------- decide idx_xt init -----------------
    if args.resample_xt:
        idx_xt = sample_xt_idx_np()
        xt_resample_enabled = True
    else:
        idx_xt = idx_xt_fixed
        xt_resample_enabled = False

    # ----------------- case resample decision logic -----------------
    amp_for_capacity = bool(use_amp_global and device.type == "cuda")

    if args.case_batch > 0:
        B_upper = min(args.case_batch, num_cases)
    else:
        B_upper = num_cases

    #    If we are allowed to use all cases (i.e., no user cap below num_cases),
    #    try full batch first. If it fits, DON'T subtract safety margin.
    can_try_full = (B_upper == num_cases)

    if can_try_full:
        if not _one_step_oom_test(num_cases, idx_xt, amp_enabled_test=amp_for_capacity):
            Bcase = num_cases
            case_resample_enabled = False
            idx_case = np.arange(num_cases, dtype=np.int64)
            print(f"[case] full batch fits: Bcase={Bcase} (no sampling, no safety margin)")
        else:
            # full doesn't fit -> find maximum <= num_cases, then apply safety margin
            Bmax_amp = find_max_Bcase(idx_xt, amp_enabled_search=amp_for_capacity, B_start=num_cases)
            Bcase = max(1, Bmax_amp - 10)
            case_resample_enabled = (Bcase < num_cases)
            idx_case = sample_case_idx_np(Bcase, epoch=0)
            print(f"[case] full batch OOM: Bmax_amp={Bmax_amp}, Bcase(train)={Bcase} (safety -10)")
    else:
        # user cap < num_cases:
        # try the cap first; if it fits, use it directly (no safety margin).
        if not _one_step_oom_test(B_upper, idx_xt, amp_enabled_test=amp_for_capacity):
            Bcase = B_upper
            case_resample_enabled = True  # still sampling, since not full dataset
            idx_case = sample_case_idx_np(Bcase, epoch=0)
            print(f"[case] capped fits: B_upper={B_upper}, Bcase(train)={Bcase} (no safety margin)")
        else:
            # cap OOM -> find max <= cap, then apply safety margin
            Bmax_amp = find_max_Bcase(idx_xt, amp_enabled_search=amp_for_capacity, B_start=B_upper)
            Bcase = max(1, Bmax_amp - 10)
            case_resample_enabled = True
            idx_case = sample_case_idx_np(Bcase, epoch=0)
            print(f"[case] capped OOM: B_upper={B_upper}, Bmax_amp={Bmax_amp}, Bcase(train)={Bcase} (safety -10)")


    print(
        f"[Train setup] Bcase={Bcase}, Pxt={Pxt}, device={device}, "
        f"input_mode={input_mode}, in_dim={in_dim}, M=({M_beta},{M_nu},{M_rho}), K={K}, "
        f"xt_resample={xt_resample_enabled}, case_resample={case_resample_enabled}"
    )

    # save normalization/config
    np.savez(
        os.path.join(path_str, "normalization_params.npz"),
        xt_mean=xt_mean,
        xt_std=xt_std,
        range_beta=np.array(args.range_beta, dtype=np.float32),
        range_nu=np.array(args.range_nu, dtype=np.float32),
        range_rho=np.array(args.range_rho, dtype=np.float32),
        target_range_beta=np.array(target_range_beta, dtype=np.float32),
        target_range_nu=np.array(target_range_nu, dtype=np.float32),
        target_range_rho=np.array(target_range_rho, dtype=np.float32),
        M_beta=np.int32(M_beta),
        M_nu=np.int32(M_nu),
        M_rho=np.int32(M_rho),
        input_mode=np.array([input_mode]),
        U_all_on_gpu=np.array([args.U_all_on_gpu], dtype=np.int32),
        act=np.array([args.act]),
        resample_xt=np.array([int(args.resample_xt)], dtype=np.int32),
        resample_xt_interval=np.int32(args.resample_xt_interval),
        resample_case_interval=np.int32(args.resample_case_interval),
        Bcase=np.int32(Bcase),
        Pxt=np.int32(Pxt),
        amp_first80_fp16=np.array([int(use_amp_global)], dtype=np.int32),
        amp_switch_iter=np.int32(switch_iter),
        xtp_extra_feat=np.array(["log1p(beta/nu)", "log1p(rho/nu)", "log1p(rho/beta)"]),
        basis=np.array([args.basis]),
        clamp_size=np.float32(args.clamp_size),
    )

    # logs
    loss_list, mse_list, nmae_list, l2re_list = [], [], [], []

    # flags for fixed sampling in last 20%
    switched_to_fp32 = False  # only to ensure switch logic runs once

    for epoch in tqdm(range(args.iter)):
        amp_enabled = bool(use_amp_global and (epoch < switch_iter) and (device.type == "cuda"))

        if (not switched_to_fp32) and (epoch >= switch_iter):
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Search upper bound in FP32:
            # Prefer to start from user cap if any, otherwise from current Bcase+10 (undo safety) but not exceed num_cases.
            if args.case_batch > 0:
                B_upper_fp32 = min(args.case_batch, num_cases)
            else:
                B_upper_fp32 = min(Bcase + 10, num_cases)

            if not _one_step_oom_test(B_upper_fp32, idx_xt, amp_enabled_test=False):
                Bcase_fp32 = B_upper_fp32
                print(f"\n[FP32 switch] cap fits: Bcase stays/sets to {Bcase_fp32} (no safety margin)")
            else:
                Bmax_fp32 = find_max_Bcase(idx_xt, amp_enabled_search=False, B_start=B_upper_fp32)
                Bcase_fp32 = max(1, Bmax_fp32 - 10)
                print(f"\n[FP32 switch] cap OOM: Bmax_fp32={Bmax_fp32}, Bcase_fp32={Bcase_fp32} (safety -10)")

            if Bcase_fp32 < Bcase:
                print(f"\n[FP32 switch] Bcase {Bcase} -> {Bcase_fp32} (Bmax_fp32={Bmax_fp32}, safety -10)")
                Bcase = Bcase_fp32
                idx_case = sample_case_idx_np(Bcase, epoch)  # batch size changed -> resample

            switched_to_fp32 = True
            print(f"\n[Switch @ iter {epoch}] Entering last 20%: AMP=OFF (FP32), sampling NOT frozen.\n")

        if xt_resample_enabled and (epoch % args.resample_xt_interval == 0):
            idx_xt = sample_xt_idx_np()

        # case resample interval: half in last 20%
        case_interval_now = args.resample_case_interval
        if epoch >= switch_iter:
            case_interval_now = max(1, args.resample_case_interval // 2)

        if case_resample_enabled and (epoch % case_interval_now == 0):
            idx_case = sample_case_idx_np(Bcase, epoch)

        # --------- train step with OOM retry (Bcase -= 5 until success) ---------
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

                        if args.loss_mode == "mse":
                            loss = torch.mean((pred - u_t) ** 2)
                            mse = loss.detach()
                            nmae = torch.tensor(0.0, device=device)
                        else:
                            loss, mse, nmae = loss_mse_nmae(
                                pred,
                                u_t,
                                eps=args.nmae_eps,
                                mix_lambda=args.mix_lambda,
                                per_case=args.per_case_loss,
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

                    if args.loss_mode == "mse":
                        loss = torch.mean((pred - u_t) ** 2)
                        mse = loss.detach()
                        nmae = torch.tensor(0.0, device=device)
                    else:
                        loss, mse, nmae = loss_mse_nmae(
                            pred,
                            u_t,
                            eps=args.nmae_eps,
                            mix_lambda=args.mix_lambda,
                            per_case=args.per_case_loss,
                            Bcase=Bb,
                            Pxt=Pxt,
                        )

                    loss.backward()
                    optimizer.step()

                # success
                break

            except torch.cuda.OutOfMemoryError:
                if device.type != "cuda":
                    raise  # non-cuda OOM shouldn't happen here; re-raise for visibility

                attempt += 1
                # best-effort cleanup
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

                new_Bcase = max(1, int(Bcase * 0.95))
                # 或 new_Bcase = max(1, Bcase - 50)
                if new_Bcase == Bcase:
                    raise RuntimeError("OOM even at Bcase=1. Need reduce xt_points/hidden_size/M/K or move U_all to CPU.")
                Bcase = new_Bcase

                # if we previously had full-batch (no sampling), now we must enable sampling
                if Bcase < num_cases:
                    case_resample_enabled = True

                # batch size changed -> resample cases
                idx_case = sample_case_idx_np(Bcase, epoch)

                print(f"[OOM] iter={epoch}, attempt={attempt}: reduce Bcase to {Bcase} and retry.")

                if attempt >= max_retries:
                    raise RuntimeError(f"Exceeded max_retries={max_retries} at iter={epoch}. Last Bcase={Bcase}.")
                continue
        # ----------------------------------------------------------------------

        loss_list.append(float(loss.item()))
        mse_list.append(float(mse.item()))
        nmae_list.append(float(nmae.item()))

        if epoch % args.eval_interval == 0:
            netW.eval()
            with torch.no_grad():
                Wt = netW(test_feat_t)
                test_pred = torch.sum(Wt * test_Phi, dim=1, keepdim=True)
                num = torch.linalg.vector_norm(test_pred - test_u_t)
                den = torch.linalg.vector_norm(test_u_t) + 1e-12
                l2re = (num / den).item()
                l2re_list.append(float(l2re))

                if epoch > 0:
                    # piecewise exponential decay, updated only at eval points
                    gamma = args.lr_gamma1 if epoch < (args.iter // 2) else args.lr_gamma2
                    for pg in optimizer.param_groups:
                        pg["lr"] *= gamma
                lr_now = optimizer.param_groups[0]["lr"]
                phase_tag = "FP32" if (epoch >= switch_iter) else "AMP"
                p = epoch / float(args.iter)
                if p < INT_STAGE1:
                    pool_tag = "int_first"
                elif p < INT_STAGE2:
                    pool_tag = f"mix_int{INT_MIX:.2f}"
                else:
                    pool_tag = "all"
                # NMAE={nmae.item():.6e},
                print(
                    f"\nIter {epoch}: "
                    f"{phase_tag}, "
                    f"Pool={pool_tag}, "
                    f"Loss={loss.item():.6e}, MSE={mse.item():.6e}, " 
                    f"L2re(test)={l2re:.6f}, LR={lr_now:.2e}"
                )

    checkpoint = {
        "epoch": args.iter,
        "model_state_dict": netW.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "xt_mean": xt_mean,
        "xt_std": xt_std,
        "M_beta": M_beta,
        "M_nu": M_nu,
        "M_rho": M_rho,
        "range_beta": args.range_beta,
        "range_nu": args.range_nu,
        "range_rho": args.range_rho,
        "target_range_beta": target_range_beta,
        "target_range_nu": target_range_nu,
        "target_range_rho": target_range_rho,
        "input_mode": input_mode,
        "in_dim": in_dim,
        "K": K,
        "U_all_on_gpu": int(args.U_all_on_gpu),
        "act": args.act,
        "N_total": int(N_total),
        "num_cases": int(num_cases),
        "interval": float(args.interval),
        "Bcase": int(Bcase),
        "Pxt": int(Pxt),
        "resample_xt": bool(args.resample_xt),
        "resample_case": bool(case_resample_enabled),
        "amp_first80_fp16": bool(use_amp_global),
        "amp_switch_iter": int(switch_iter),
        "basis": args.basis,
        "clamp_size": float(args.clamp_size),
    }
    torch.save(checkpoint, os.path.join(path_str, "checkpoint.pth"))

    pd.DataFrame({"loss": loss_list, "mse": mse_list, "nmae": nmae_list}).to_csv(os.path.join(path_str, "loss.csv"), index=False)
    pd.DataFrame({"l2re": l2re_list}).to_csv(os.path.join(path_str, "l2re.csv"), index=False)

    print("Saved to:", path_str)


if __name__ == "__main__":
    main()