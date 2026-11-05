"""
Bayesian Optimization Engine for Antenna Design using BoTorch and PyTorch.
"""
import torch
import numpy as np
import pandas as pd
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from src.metrics import analyze_s11


class BayesianOptimizer:
    def __init__(self, param_bounds: dict, driver, target_freq: float = 2.40, mode: str = "balanced", constraints: dict = None):
        """
        Initializes the Bayesian Optimization loop.
        """
        self.param_names = list(param_bounds.keys())
        self.bounds = torch.tensor(
            [
                [param_bounds[p][0] for p in self.param_names],
                [param_bounds[p][1] for p in self.param_names],
            ],
            dtype=torch.float64,
        )
        self.driver = driver
        self.target_freq = target_freq
        self.mode = mode
        self.constraints = constraints or {}

        self.history = []
        self.best_s11_curve = None
        self.best_params = None
        self.best_score = float("inf")

    def _normalize(self, X):
        return (X - self.bounds[0]) / (self.bounds[1] - self.bounds[0])

    def _unnormalize(self, X_norm):
        return self.bounds[0] + X_norm * (self.bounds[1] - self.bounds[0])

    def _evaluate_point(self, param_values):
        param_dict = {name: float(val) for name, val in zip(self.param_names, param_values)}
        freqs, s11_db = self.driver.run_simulation(param_dict)
        metrics = analyze_s11(
            freqs,
            s11_db,
            target_freq=self.target_freq,
            mode=self.mode,
            constraints=self.constraints,
        )

        score = metrics["objective_score"]
        if score < self.best_score:
            self.best_score = score
            self.best_params = param_dict
            self.best_s11_curve = (freqs, s11_db)

        record = {**param_dict, **metrics}
        self.history.append(record)
        return score, freqs, s11_db

    def run(self, n_init: int = 5, n_iter: int = 15):
        print(f"=== Starting Bayesian Optimization ({n_init} Init + {n_iter} Active Iterations) ===")
        print(f"[*] Strategy Mode: '{self.mode}' | Target: {self.target_freq} GHz\n")

        # 1. Initial Design of Experiments
        dim = len(self.param_names)
        init_X_norm = torch.rand(n_init, dim, dtype=torch.float64)
        init_X = self._unnormalize(init_X_norm)

        train_Y = []
        for i, x in enumerate(init_X):
            print(f"[Init {i+1}/{n_init}] Evaluating initial configuration...")
            score, _, _ = self._evaluate_point(x)
            train_Y.append(-score)

        train_X_norm = init_X_norm
        train_Y = torch.tensor(train_Y, dtype=torch.float64).unsqueeze(-1)

        # 2. Active Learning Loop
        unit_bounds = torch.stack([torch.zeros(dim, dtype=torch.float64), torch.ones(dim, dtype=torch.float64)])

        for it in range(n_iter):
            print(f"\n[Iteration {it+1}/{n_iter}]")
            gp = SingleTaskGP(train_X_norm, train_Y)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)

            best_y = train_Y.max()
            qEI = qLogExpectedImprovement(model=gp, best_f=best_y)

            candidate_norm, _ = optimize_acqf(
                qEI,
                bounds=unit_bounds,
                q=1,
                num_restarts=5,
                raw_samples=20,
            )

            candidate_raw = self._unnormalize(candidate_norm[0])
            score, _, _ = self._evaluate_point(candidate_raw)

            train_X_norm = torch.cat([train_X_norm, candidate_norm], dim=0)
            train_Y = torch.cat([train_Y, torch.tensor([[-score]], dtype=torch.float64)], dim=0)

            last = self.history[-1]
            print(f"   -> Resonance: {last['resonance_freq_ghz']:.3f} GHz, Min S11: {last['min_s11_db']:.2f} dB, BW: {last['bandwidth_mhz']:.1f} MHz, Score: {score:.2f}")

        df_results = pd.DataFrame(self.history)
        print("\n=== Optimization Finished ===")
        print(f"Best Score: {self.best_score:.2f}")
        print(f"Best Parameters: {self.best_params}")
        return df_results, self.best_params, self.best_s11_curve