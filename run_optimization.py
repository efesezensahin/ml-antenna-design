import argparse
import yaml
import matplotlib.pyplot as plt
from src.cst_driver import CSTDriver
from src.mock_driver import MockAntennaDriver
from src.optimizer import BayesianOptimizer


def main():
    parser = argparse.ArgumentParser(
        description="Machine Learning Active Learning Framework for RF Antenna Optimization"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="mock",
        choices=["mock", "cst"],
        help="'mock' (license-free analytical physics) or 'cst' (live CST Studio Suite solver)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/patch_2p4ghz.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        choices=["balanced", "max_bandwidth", "deep_s11"],
        help="Override objective strategy (defaults to YAML setting if omitted)",
    )
    parser.add_argument(
        "--n_init",
        type=int,
        default=5,
        help="Number of initial design exploration samples",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=15,
        help="Number of active Bayesian optimization iterations",
    )
    args = parser.parse_args()

    # 1. Load Configuration
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    param_bounds = {p: (cfg["min"], cfg["max"]) for p, cfg in config["parameters"].items()}
    target_freq = config["objectives"].get("target_frequency_ghz", 2.40)
    strategy = args.strategy or config["objectives"].get("mode", "balanced")
    constraints = config.get("constraints", {})

    # 2. Select Driver
    if args.mode == "cst":
        print("[MODE] Live CST Studio Suite driver active.")
        cst_path = config["project"]["cst_project_path"]
        driver = CSTDriver(cst_path)
    else:
        print("[MODE] License-free Mock driver active (analytical cavity model).")
        driver = MockAntennaDriver()

    # 3. Launch Optimizer
    optimizer = BayesianOptimizer(
        param_bounds=param_bounds,
        driver=driver,
        target_freq=target_freq,
        mode=strategy,
        constraints=constraints,
    )
    df_results, best_params, best_s11_curve = optimizer.run(n_init=args.n_init, n_iter=args.n_iter)

    # 4. Save History
    df_results.to_csv("optimization_history.csv", index=False)
    print("\n[+] Results saved to 'optimization_history.csv'.")

    # 5. Plot and Save Visualizations
    freqs, s11_db = best_s11_curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Convergence plot
    best_so_far = df_results["objective_score"].cummin()
    ax1.plot(df_results.index + 1, df_results["objective_score"], "o-", color="gray", alpha=0.5, label="Step Score")
    ax1.plot(df_results.index + 1, best_so_far, "r-", lw=2.5, label="Best Score")
    ax1.axvline(args.n_init, color="black", linestyle="--", label="Active Loop Start")
    ax1.set_xlabel("Simulation Number")
    ax1.set_ylabel("Objective Score (Lower is Better)")
    ax1.set_title(f"Optimization Convergence ({strategy.capitalize()} Mode)")
    ax1.grid(True)
    ax1.legend()

    # Optimal S11 plot
    ax2.plot(freqs, s11_db, color="navy", lw=2, label="Optimal S11")
    ax2.axhline(-10, color="red", linestyle="--", label="-10 dB Threshold")
    ax2.axvline(target_freq, color="green", linestyle=":", label=f"Target: {target_freq} GHz")
    ax2.set_xlabel("Frequency (GHz)")
    ax2.set_ylabel("S11 (dB)")
    ax2.set_title(f"Optimal S11 Response\n(anty={best_params['anty']:.2f}mm, antx={best_params['antx']:.2f}mm)")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("optimization_results.png", dpi=150)
    print("[+] Plot saved to 'optimization_results.png'!\n")
    plt.show()


if __name__ == "__main__":
    main()