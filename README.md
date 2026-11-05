# 📡 Active Learning & Bayesian Optimization for Automated Antenna Design

<img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python">
<img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch">
<img src="https://img.shields.io/badge/BoTorch-0.10+-red.svg" alt="BoTorch">
<img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">

An open-source, sample-efficient **Active Learning (Bayesian Optimization)** framework designed to automate radio frequency (RF) antenna design. The framework bridges modern machine learning (Meta's BoTorch & PyTorch) with 3D full-wave electromagnetic (EM) solvers (Dassault Systèmes CST Studio Suite) to optimize multi-parameter antenna geometries in a fraction of the traditional simulation budget.

---

## 🌟 Highlights

* **Sample-Efficient AI Optimization:** Replaces brute-force grid sweeps ($10^4+$ simulations) with Gaussian Process (GP) active learning, converging in just **15–20 simulations**.
* **Dual-Backend Architecture:**
  * **Mock Mode (Zero-License):** Built-in analytical physics engine (cavity & transmission line model) allowing any researcher on Linux, macOS, or Windows to reproduce the optimization loop in ~10 seconds without proprietary EDA tools.
  * **Live CST Studio Mode:** Real-time Windows COM automation for **CST Studio Suite (2019–2026)**.
* **Direct SQLite Binary Decoding:** Completely bypasses slow CST GUI ASCII file exports by directly reading and decoding IEEE 754 float/complex S-parameter blobs from `Result/Storage.sdb` in milliseconds.
* **Constraint-Aware Multi-Strategy Engine:** Supports customizable optimization objectives (`balanced`, `max_bandwidth`, `deep_s11`) with strict frequency drift and return loss barrier penalties.

---

## 📊 Benchmark Results (Live CST Studio Suite 2019)

We validated the pipeline on a 2.40 GHz inset-fed microstrip patch antenna on an FR-4 substrate ($\epsilon_r = 4.3$, $h = 1.5\text{ mm}$):

| Parameter / Metric | Baseline (Manual Design) | BoTorch Optimized | Engineering Impact |
| :--- | :--- | :--- | :--- |
| **Resonant Frequency ($f_0$)** | 2.392 GHz | **2.401 GHz** | **8× accuracy improvement** (only 1 MHz / 0.04% error) |
| **Return Loss ($S_{11}$)** | -55.80 dB | **-40.65 dB** | **99.991% radiated power** (reflection virtually zero) |
| **Objective Score** | -41.90 | **-42.60** | Optimal joint convergence |
| `anty` (Patch Length) | 30.20 mm | **30.09 mm** | Fine-tuned by 0.11 mm to shift resonance +9 MHz |
| `antx` (Patch Width) | 47.00 mm | **40.00 mm** | Compact aperture |
| `insy` (Inset Depth) | 7.16 mm | **4.00 mm** | Matched to 50 $\Omega$ input feed |
| `insx` (Inset Notch Gap) | 1.50 mm | **2.50 mm** | Capacitive reactance tuned |

### Convergence & Return Loss Response
[optimization_results.png]

---

## ⚡ Quickstart

### 1. Installation
```bash
# Clone the repository
git clone [https://github.com/your-username/ml-antenna-design.git](https://github.com/your-username/ml-antenna-design.git)
cd ml-antenna-design

# Create and activate virtual environment
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
