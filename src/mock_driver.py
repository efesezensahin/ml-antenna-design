import numpy as np


class MockAntennaDriver:
    """
    Synthetic antenna driver emulating microstrip patch physics using an
    analytical (cavity / transmission-line) model for systems without CST Studio Suite.
    Provides an identical API interface to CSTDriver.
    """

    def __init__(self, f_start=1.0, f_stop=4.0, n_points=1001):
        self.freqs = np.linspace(f_start, f_stop, n_points, dtype=np.float32)

    def run_simulation(self, param_dict):
        """
        Runs the analytical simulation for the given geometric parameters.

        Args:
            param_dict (dict): Antenna dimensions, e.g.,
                {'anty': 30.5, 'antx': 47.0, 'insy': 7.16, 'insx': 1.5}

        Returns:
            tuple[np.ndarray, np.ndarray]: (frequencies, S11 in dB)
        """
        anty = float(param_dict.get("anty", 30.2))
        antx = float(param_dict.get("antx", 47.0))
        insy = float(param_dict.get("insy", 7.16))
        insx = float(param_dict.get("insx", 1.5))

        # 1. Physical Resonant Frequency (Cavity / Transmission Line Model):
        # Increasing patch length (anty) lowers the resonant frequency.
        # Length ~30.2 mm corresponds to 2.40 GHz resonance.
        c = 300.0  # Speed of light in mm * GHz
        eps_eff = 3.8  # Effective dielectric constant for FR-4 substrate
        delta_l = 0.8  # Length extension due to fringing fields

        # Effective length and fundamental resonance
        l_eff = anty + 2 * delta_l
        f0 = c / (2 * l_eff * np.sqrt(eps_eff))

        # 2. Impedance Matching and Return Loss Depth:
        # Inset depth (insy = 7.16 mm) achieves nominal 50 Ohm input match.
        # Impedance mismatch degrades S11 depth (from -55 dB up to -3 dB).
        matching_error = abs(insy - 7.16) / 3.0 + abs(insx - 1.5) / 1.5
        min_depth = -55.0 + (matching_error ** 2) * 25.0
        min_depth = np.clip(min_depth, -55.0, -3.0)

        # 3. Bandwidth Factor (wider patch increases radiation conductance and bandwidth)
        q_factor = 35.0 * (47.0 / max(antx, 20.0))
        bw_ghz = f0 / q_factor

        # 4. Synthetic Lorentzian S11 Curve Synthesis
        gamma = bw_ghz / 2.0
        lorentzian = (gamma ** 2) / ((self.freqs - f0) ** 2 + gamma ** 2)
        s11_db = min_depth * lorentzian

        # Add secondary higher-order harmonic mode around 3.15 GHz (emulating real CST behavior)
        f_harmonic = f0 * 1.31
        harm_lorentzian = (0.04 ** 2) / ((self.freqs - f_harmonic) ** 2 + 0.04 ** 2)
        s11_db += -13.0 * harm_lorentzian

        # Clamp baseline reflection to 0 dB ceiling
        s11_db = np.clip(s11_db, -60.0, 0.0)

        return self.freqs, s11_db