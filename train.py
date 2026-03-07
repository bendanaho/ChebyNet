from __future__ import annotations

import argparse
from typing import List, Tuple

from pde_specs import get_pde_spec
from trainer import TrainConfig, train_supervised


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    p.add_argument("--pde", type=str, default=None, help="Which PDE spec to use.")
    p.add_argument("--basis", type=str, default="cheb", choices=["cheb", "polyn"])

    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--hidden_size", type=int, default=100)
    p.add_argument("--act", type=str, default="gelu", choices=["tanh", "gelu", "gelu_tanh", "silu"])

    p.add_argument(
        "--Ms",
        type=int,
        nargs="*",
        default=None,
        help="Degrees per parameter dim (len=param_dim). Example: --Ms 6 6 6",
    )

    p.add_argument("--pretrain", action="store_true")
    p.add_argument(
        "--target_param_ranges",
        type=float,
        nargs="*",
        default=None,
        help="Flattened target ranges for unit scaling: a1 b1 a2 b2 ...",
    )

    p.add_argument("--input_mode", type=str, default="xtp", choices=["auto", "xt", "xtp"])
    p.add_argument("--auto_threshold", type=float, default=10.0)
    p.add_argument(
        "--use_ratio_feats",
        action="store_true",
        help="When input_mode=xtp: include log1p(pi/pj) ratio features. Default: OFF.",
    )

    p.add_argument("--device", type=str, default="0")
    p.add_argument("--seed", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--iter", type=int, default=100000)
    p.add_argument("--lr_gamma1", type=float, default=0.99)
    p.add_argument("--lr_gamma2", type=float, default=0.987)
    p.add_argument("--eval_interval", type=int, default=500)

    p.add_argument("--loss_mode", type=str, default="mse", choices=["mse", "mix"])
    p.add_argument("--mix_lambda", type=float, default=0.95)
    p.add_argument("--nmae_eps", type=float, default=1e-3)
    p.add_argument("--per_case_loss", action="store_true")

    p.add_argument("--xt_points", type=int, default=2000)

    p.add_argument("--case_batch", type=int, default=0)
    p.add_argument("--resample_case_interval", type=int, default=1000)

    p.add_argument("--resample_xt", action="store_true", default=False)
    p.add_argument("--resample_xt_interval", type=int, default=5000)

    p.add_argument("--test_params", type=float, nargs="*", default=None, help="Test parameters (len=param_dim).")

    p.add_argument("--data_dir", type=str, default="")
    p.add_argument("--save_dir", type=str, default="runs")

    p.add_argument("--warm_start_path", type=str, default="")
    p.add_argument("--warm_start_strict", action="store_true")

    p.add_argument("--U_all_on_gpu", type=int, default=1)

    # AMP 80% strategy switch
    p.add_argument("--amp80", dest="amp80", action="store_true", help="Enable AMP for first 80% iters then FP32.")
    p.add_argument("--no-amp80", dest="amp80", action="store_false", help="Disable AMP80 strategy (always FP32).")
    p.set_defaults(amp80=True)

    # normalize_io only controls net input; Phi always uses unit([-1,1]) params for both cheb/polyn
    p.add_argument(
        "--normalize_io",
        type=int,
        default=1,
        choices=[0, 1],
        help="0: raw coords + raw params into net input (no ratio feats). "
        "1: original net-input normalization behavior. "
        "Basis Phi (cheb/polyn) always uses unit([-1,1]) params.",
    )

    return p


def _parse_flat_ranges(flat: List[float], d: int, name: str) -> Tuple[Tuple[float, float], ...]:
    if flat is None:
        raise ValueError(f"{name} is None")
    if len(flat) != 2 * d:
        raise ValueError(f"{name} must have length {2*d} (got {len(flat)}) for param_dim={d}")
    pairs = []
    for i in range(d):
        a = float(flat[2 * i])
        b = float(flat[2 * i + 1])
        pairs.append((a, b))
    return tuple(pairs)


def main():
    parser = build_parser()
    args = parser.parse_args()

    spec = get_pde_spec(args.pde)
    d = spec.param_dim

    if args.Ms is not None and len(args.Ms) > 0:
        if len(args.Ms) != d:
            raise ValueError(f"--Ms must have length {d} for PDE '{spec.name}' (param_dim={d}).")
        Ms = tuple(int(x) for x in args.Ms)
    else:
        Ms = spec.default_Ms
        if len(Ms) != d:
            raise ValueError(f"Spec default_Ms length must equal param_dim={d} (got {len(Ms)}).")

    if args.pretrain:
        if args.target_param_ranges is None or len(args.target_param_ranges) == 0:
            raise ValueError("--pretrain requires --target_param_ranges")
        target_param_ranges = _parse_flat_ranges(args.target_param_ranges, d, "--target_param_ranges")
    else:
        target_param_ranges = None

    if args.test_params is not None and len(args.test_params) > 0:
        if len(args.test_params) != d:
            raise ValueError(f"--test_params must have length {d}")
        test_params = tuple(float(x) for x in args.test_params)
    else:
        test_params = tuple(float(x) for x in spec.default_test_params)

    data_dir = args.data_dir if args.data_dir else spec.default_data_dir

    cfg = TrainConfig(
        basis=args.basis,
        layers=args.layers,
        hidden_size=args.hidden_size,
        act=args.act,
        Ms=Ms,
        pretrain=bool(args.pretrain),
        target_param_ranges=target_param_ranges,
        input_mode=args.input_mode,
        auto_threshold=args.auto_threshold,
        use_ratio_feats=bool(args.use_ratio_feats),
        device=args.device,
        seed=args.seed,
        lr=args.lr,
        weight_decay=args.weight_decay,
        iters=args.iter,
        lr_gamma1=args.lr_gamma1,
        lr_gamma2=args.lr_gamma2,
        eval_interval=args.eval_interval,
        loss_mode=args.loss_mode,
        mix_lambda=args.mix_lambda,
        nmae_eps=args.nmae_eps,
        per_case_loss=bool(args.per_case_loss),
        xt_points=args.xt_points,
        case_batch=args.case_batch,
        resample_case_interval=args.resample_case_interval,
        resample_xt=bool(args.resample_xt),
        resample_xt_interval=args.resample_xt_interval,
        test_params=test_params,
        data_dir=data_dir,
        save_dir=args.save_dir,
        warm_start_path=args.warm_start_path,
        warm_start_strict=bool(args.warm_start_strict),
        U_all_on_gpu=args.U_all_on_gpu,
        amp80=bool(args.amp80),
        normalize_io=int(args.normalize_io),
    )

    train_supervised(spec, cfg)


if __name__ == "__main__":
    main()