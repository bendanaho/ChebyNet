from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np

from data import (
    load_case_cdr,
    load_solution_only_cdr,
    load_case_convection,
    load_solution_only_convection,
    load_case_heat,
    load_solution_only_heat,
    load_case_reaction,
    load_solution_only_reaction,
    load_case_burgers,
    load_solution_only_burgers,
    load_case_allen_cahn_1d,
    load_solution_only_allen_cahn_1d,
    load_case_allen_cahn_2d,
    load_solution_only_allen_cahn_2d,
    load_case_ns_2d,
    load_solution_only_ns_2d,
)


@dataclass(frozen=True)
class PDESpec:
    name: str
    coord_dim: int  # xt => 2, xyt => 3

    param_dim: int
    param_names: Tuple[str, ...]

    default_Ms: Tuple[int, ...]

    # training cases on disk
    train_param_cases: np.ndarray  # (num_train_cases, param_dim)

    # one default parameter used during training-time eval display
    default_test_params: Tuple[float, ...]

    # final test cases (Table 5 testing column)
    final_test_param_cases: np.ndarray  # (num_final_test_cases, param_dim)

    load_case: Callable[..., tuple[np.ndarray, np.ndarray]]
    load_solution_only: Callable[..., np.ndarray]

    default_data_dir: str


def _col(vs: list[float]) -> np.ndarray:
    return np.array(vs, dtype=np.float32).reshape(-1, 1)


def _pairs(pairs: list[tuple[float, ...]]) -> np.ndarray:
    return np.array(pairs, dtype=np.float32)


def get_pde_spec(name: str) -> PDESpec:
    name = name.lower().strip()

    # ----------------- CDR -----------------
    if name in {"cdr", "cdr3"}:
        # TODO: Replace with your real case list if your disk contains more combinations.
        train_cases = np.array(
            [
                [1.0, 1.0, 1.0],
                [3.0, 3.0, 3.0],
                [5.0, 5.0, 5.0],
                [7.0, 7.0, 7.0],
                [9.0, 9.0, 9.0],
            ],
            dtype=np.float32,
        )
        # No Table 5 row provided for CDR in your screenshot, keep a safe default:
        final_tests = np.array([[3.25, 3.25, 3.25]], dtype=np.float32)
        return PDESpec(
            name="cdr",
            coord_dim=2,
            param_dim=3,
            param_names=("beta", "nu", "rho"),
            default_Ms=(6, 6, 6),
            train_param_cases=train_cases,
            default_test_params=(3.25, 3.25, 3.25),
            final_test_param_cases=final_tests,
            load_case=load_case_cdr,
            load_solution_only=load_solution_only_cdr,
            default_data_dir="ground_truth/Convection-Diffusion-Reaction",
        )

    # ----------------- Convection -----------------
    if name in {"convection", "conv"}:
        # training: 1,3,...,19
        train_betas = list(np.arange(1.0, 20.0, 2.0))
        # testing (Table 5): 6,8,10,14,18
        test_betas = [6.0, 8.0, 10.0, 14.0, 18.0]
        return PDESpec(
            name="convection",
            coord_dim=2,
            param_dim=1,
            param_names=("beta",),
            default_Ms=(12,),
            train_param_cases=_col(train_betas),
            default_test_params=(6.0,),
            final_test_param_cases=_col(test_betas),
            load_case=load_case_convection,
            load_solution_only=load_solution_only_convection,
            default_data_dir="ground_truth/Convection",
        )

    if name in {"convection_extrap", "conv_extrap", "convection-extrap"}:
        train_betas = list(np.arange(1.0, 20.0, 2.0))
        # extrap testing (Table 5): 30,40,60
        test_betas = [30.0, 40.0, 60.0]
        return PDESpec(
            name="convection_extrap",
            coord_dim=2,
            param_dim=1,
            param_names=("beta",),
            default_Ms=(12,),
            train_param_cases=_col(train_betas),
            default_test_params=(30.0,),
            final_test_param_cases=_col(test_betas),
            load_case=load_case_convection,
            load_solution_only=load_solution_only_convection,
            default_data_dir="ground_truth/Convection",
        )

    # ----------------- Heat -----------------
    if name in {"heat"}:
        # training alpha: 0.5..9.5 step 1
        train_alphas = list(np.arange(0.5, 10.0, 1.0))
        # testing (Table 5): 4,5,6,7,8
        test_alphas = [4.0, 5.0, 6.0, 7.0, 8.0]
        return PDESpec(
            name="heat",
            coord_dim=2,
            param_dim=1,
            param_names=("alpha",),
            default_Ms=(12,),
            train_param_cases=_col(train_alphas),
            default_test_params=(4.0,),
            final_test_param_cases=_col(test_alphas),
            load_case=load_case_heat,
            load_solution_only=load_solution_only_heat,
            default_data_dir="ground_truth/Heat",
        )

    # ----------------- Reaction -----------------
    if name in {"reaction"}:
        train_rhos = list(np.arange(1.0, 11.0, 1.0))
        # testing (Table 5): 4.5,5.5,6.5,7.5,8.5,9.5
        test_rhos = [4.5, 6.5, 7.5, 8.5, 9.5]
        return PDESpec(
            name="reaction",
            coord_dim=2,
            param_dim=1,
            param_names=("rho",),
            default_Ms=(12,),
            train_param_cases=_col(train_rhos),
            default_test_params=(4.5,),
            final_test_param_cases=_col(test_rhos),
            load_case=load_case_reaction,
            load_solution_only=load_solution_only_reaction,
            default_data_dir="ground_truth/Reaction",
        )

    if name in {"reaction_extrap", "reaction-extrap"}:
        train_rhos = list(np.arange(1.0, 11.0, 1.0))
        # extrap testing (Table 5): 20,30
        test_rhos = [20.0, 30.0]
        return PDESpec(
            name="reaction_extrap",
            coord_dim=2,
            param_dim=1,
            param_names=("rho",),
            default_Ms=(12,),
            train_param_cases=_col(train_rhos),
            default_test_params=(20.0,),
            final_test_param_cases=_col(test_rhos),
            load_case=load_case_reaction,
            load_solution_only=load_solution_only_reaction,
            default_data_dir="ground_truth/Reaction",
        )

    # ----------------- Burgers -----------------
    if name in {"burgers"}:
        # training (Table 5): 0.001,0.002,0.005,0.105,0.205,...,0.905
        train_nus = [0.001, 0.002, 0.005] + [round(x, 3) for x in np.arange(0.105, 0.906, 0.1)]
        # testing (Table 5): 0.0015,0.0025,0.003,0.004,0.1
        test_nus = [0.0015, 0.0025, 0.003, 0.004, 0.1]
        return PDESpec(
            name="burgers",
            coord_dim=2,
            param_dim=1,
            param_names=("nu",),
            default_Ms=(12,),
            train_param_cases=_col(train_nus),
            default_test_params=(0.0015,),
            final_test_param_cases=_col(test_nus),
            load_case=load_case_burgers,
            load_solution_only=load_solution_only_burgers,
            default_data_dir="ground_truth/Burgers",
        )

    if name in {"burgers_extrap", "burgers-extrap"}:
        # training (Table 5): 0.205,0.305,...,0.905
        train_nus = [round(x, 3) for x in np.arange(0.205, 0.906, 0.1)]
        # extrap testing (Table 5): 0.1,0.155
        test_nus = [0.1, 0.155]
        return PDESpec(
            name="burgers_extrap",
            coord_dim=2,
            param_dim=1,
            param_names=("nu",),
            default_Ms=(12,),
            train_param_cases=_col(train_nus),
            default_test_params=(0.1,),
            final_test_param_cases=_col(test_nus),
            load_case=load_case_burgers,
            load_solution_only=load_solution_only_burgers,
            default_data_dir="ground_truth/Burgers",
        )

    # ----------------- Allen–Cahn 1D -----------------
    if name in {"allen-cahn", "allen_cahn", "ac1d"}:
        # training grid (Table 5):
        # lambda = 1e-4, 3e-4, ..., 9e-4
        # epsilon = 1,1.5,2,...,4.5
        lams = [1e-4, 3e-4, 5e-4, 7e-4, 9e-4]
        epss = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
        train_pairs = np.array([[lam, eps] for lam in lams for eps in epss], dtype=np.float32)

        # testing (Table 5) explicit pairs:
        test_pairs = [
            (1e-4, 4.3),
            (2e-4, 3.0),
            (4e-4, 3.9),
            (8e-4, 4.3),
            (5.5e-4, 3.0),
        ]
        return PDESpec(
            name="allen_cahn",
            coord_dim=2,
            param_dim=2,
            param_names=("lambda", "epsilon"),
            default_Ms=(8, 8),
            train_param_cases=train_pairs,
            default_test_params=(1e-4, 4.3),
            final_test_param_cases=_pairs(test_pairs),
            load_case=load_case_allen_cahn_1d,
            load_solution_only=load_solution_only_allen_cahn_1d,
            default_data_dir="ground_truth/Allen-Cahn",
        )

    # ----------------- Allen–Cahn 2D+T -----------------
    if name in {"allen-cahn-2d", "allen_cahn_2d", "ac2d"}:
        lams = [1e-4, 3e-4, 5e-4, 7e-4, 9e-4]
        epss = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
        train_pairs = np.array([[lam, eps] for lam in lams for eps in epss], dtype=np.float32)

        test_pairs = [
            (1e-4, 4.3),
            (2e-4, 3.0),
            (4e-4, 3.9),
            (8e-4, 4.3),
            (5.5e-4, 3.0),
        ]
        return PDESpec(
            name="allen_cahn_2d",
            coord_dim=3,
            param_dim=2,
            param_names=("lambda", "epsilon"),
            default_Ms=(8, 8),
            train_param_cases=train_pairs,
            default_test_params=(1e-4, 4.3),
            final_test_param_cases=_pairs(test_pairs),
            load_case=load_case_allen_cahn_2d,
            load_solution_only=load_solution_only_allen_cahn_2d,
            default_data_dir="ground_truth/2D-Allen-Cahn",
        )

    # ----------------- 2D+T Navier–Stokes -----------------
    # Table 5 training: 1/nu = 100,200,...,1000 => nu = 1/100 ... 1/1000
    # Table 5 testing: 1/nu = 150,350,550,750,950
    if name in {"navier-stokes-2d", "navier_stokes_2d", "ns2d", "2d_ns"}:
        invs_train = list(range(100, 1001, 100))
        train_visc = [1.0 / float(v) for v in invs_train]

        invs_test = [150, 350, 550, 750, 950]
        test_visc = [1.0 / float(v) for v in invs_test]

        return PDESpec(
            name="navier_stokes_2d",
            coord_dim=3,
            param_dim=1,
            param_names=("viscosity",),
            default_Ms=(10,),
            train_param_cases=_col(train_visc),
            default_test_params=(1.0 / 150.0,),
            final_test_param_cases=_col(test_visc),
            load_case=load_case_ns_2d,
            load_solution_only=load_solution_only_ns_2d,
            default_data_dir="ground_truth/2D-Navier–Stokes",
        )

    raise NotImplementedError(f"PDE spec '{name}' not recognized.")