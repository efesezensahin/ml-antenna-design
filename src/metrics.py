"""
RF Antenna Performance Metrics & Multi-Scenario Objective Evaluation.
Computes return loss, resonant frequency, bandwidth, and constraint-aware objective scores.
"""
import numpy as np


def analyze_s11(freqs: np.ndarray, s11_db: np.ndarray, target_freq: float = 2.40, mode: str = "balanced", constraints: dict = None) -> dict:
    """
    Extracts fundamental RF metrics from the S11 curve and computes a constraint-aware objective score.

    Args:
        freqs (np.ndarray): Frequency vector in GHz.
        s11_db (np.ndarray): Reflection coefficient (S11) in dB.
        target_freq (float): Target resonant frequency in GHz.
        mode (str): Optimization strategy ('balanced', 'max_bandwidth', 'deep_s11').
        constraints (dict): Allowed limits, e.g. {'max_freq_drift_mhz': 25.0, 'min_acceptable_s11_db': -10.0}

    Returns:
        dict: Extracted metrics including resonance frequency, min S11, bandwidth, and scalar objective score.
    """
    min_idx = np.argmin(s11_db)
    res_freq = float(freqs[min_idx])
    min_s11 = float(s11_db[min_idx])
    freq_drift_mhz = abs(res_freq - target_freq) * 1000.0

    # Calculate -10 dB impedance bandwidth
    below_10db_mask = s11_db <= -10.0
    if np.any(below_10db_mask):
        matched_freqs = freqs[below_10db_mask]
        bandwidth_mhz = float((matched_freqs[-1] - matched_freqs[0]) * 1000.0)
    else:
        bandwidth_mhz = 0.0

    # Constraint Checking (Safety Barriers)
    if constraints is None:
        constraints = {"max_freq_drift_mhz": 25.0, "min_acceptable_s11_db": -10.0}

    penalty = 0.0
    max_drift = constraints.get("max_freq_drift_mhz", 25.0)
    min_s11_req = constraints.get("min_acceptable_s11_db", -10.0)

    # 1. Frequency constraint penalty
    if freq_drift_mhz > max_drift:
        penalty += (freq_drift_mhz - max_drift) * 5.0

    # 2. S11 matching constraint penalty
    if min_s11 > min_s11_req:
        penalty += (min_s11 - min_s11_req) * 10.0

    # Objective Score Calculation (Lower is Better)
    if mode == "balanced":
        # Balances frequency alignment and reflection depth
        base_score = min_s11 + (freq_drift_mhz * 0.1)
    elif mode == "max_bandwidth":
        # Maximizes bandwidth (negated) while enforcing matching constraints
        base_score = -bandwidth_mhz if min_s11 <= min_s11_req else 0.0
    elif mode == "deep_s11":
        # Pushes S11 depth to the maximum while keeping frequency locked
        base_score = min_s11
    else:
        raise ValueError(f"Unknown objective mode: {mode}")

    total_score = base_score + penalty

    return {
        "resonance_freq_ghz": res_freq,
        "min_s11_db": min_s11,
        "bandwidth_mhz": bandwidth_mhz,
        "freq_drift_mhz": freq_drift_mhz,
        "objective_score": float(total_score),
    }