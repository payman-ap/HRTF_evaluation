import numpy as np

# Relative path definition
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))

# Import your customized SNR functions
from hrtf_eval_py.analysis.snr_methods import calculate_snr, snr_frequency_domain


def compute_snr_matrix(signals, noise, method="time"):
    # Squeeze noise and validate it's a 1D vector
    noise = np.squeeze(noise)
    if noise.ndim != 1:
        raise ValueError("Noise must be a 1D array.")

    # Ensure signals is at least 2D
    signals = np.atleast_2d(signals)

    # If signals shape is (n_dirs, n_samples), transpose it to (n_samples, n_dirs)
    if signals.shape[0] == noise.shape[0] and signals.shape[1] != noise.shape[0]:
        pass  # already (n_samples, n_dirs)
    elif signals.shape[1] == noise.shape[0] and signals.shape[0] != noise.shape[0]:
        signals = signals.T
    elif noise.shape[0] not in signals.shape:
        raise ValueError(
            f"Noise length {noise.shape[0]} does not match any signal dimension {signals.shape}."
        )

    n_samples, n_dirs = signals.shape
    snr_values = []

    for i in range(n_dirs):
        sig = signals[:, i]
        nse = noise

        if method == "time":
            snr = calculate_snr(sig, nse)
        elif method == "frequency":
            snr = snr_frequency_domain(sig, nse)
        elif method == "none":
            snr = None
        else:
            raise ValueError(f"Unknown method: {method}")

        snr_values.append(snr)

    return np.array(snr_values)




if __name__ == "__main__":
    # Synthetic test: small HRIR matrix with added noise
    np.random.seed(0)  # for reproducibility

    # Simulated HRIRs (2 directions, 5 samples each)
    hrir_signals = np.array([
        [0.1, 0.2, 0.3, 0.2, 0.1],
        [0.3, 0.4, 0.5, 0.4, 0.3]
    ])

    # Simulated noise: very low amplitude noise
    noise_floor = np.random.normal(0, 1e-4, size=(1, 5))

    snr_time = compute_snr_matrix(hrir_signals.T, noise_floor.T, method="time")
    snr_freq = compute_snr_matrix(hrir_signals.T, noise_floor.T, method="frequency")

    print("SNR (time domain):", snr_time)
    print("SNR (frequency domain):", snr_freq)
