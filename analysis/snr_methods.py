import numpy as np
from scipy.signal import stft
from scipy.fft import fft
from scipy.signal import butter, sosfilt
from scipy.signal import welch
import warnings

# for importing from hrtf_eval_py package
import sys
import os
# Go up 2 levels: from /hrtf_eval/notebooks to /my_project
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from hrtf_eval_py.utils.get_frequency_bands import get_bark_bands, get_octave_bands
from hrtf_eval_py.analysis.spectrum import real_fft_mag
from hrtf_eval_py.utils.bp_filter import bp_filter


def calculate_snr(signal, noise):
    """Time-domain energy-based SNR in dB"""
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return np.inf
    return 10 * np.log10(signal_power / noise_power)



def snr_frequency_domain(signal, noise):
    """Frequency-domain SNR (simple FFT-based)"""
    S = np.abs(fft(signal))
    N = np.abs(fft(noise))
    S_power = S ** 2
    N_power = N ** 2
    snr_freq = 10 * np.log10(np.sum(S_power) / np.sum(N_power))
    return snr_freq



import numpy as np
from numpy.fft import rfft, rfftfreq, fft

# def snr_frequency_domain(signal, noise):
#     """Frequency-domain SNR (simple FFT-based)"""
#     S = np.abs(fft(signal))
#     N = np.abs(fft(noise))
#     S_power = S ** 2
#     N_power = N ** 2
#     if np.sum(N_power) == 0:
#         return np.inf
#     snr_freq = 10 * np.log10(np.sum(S_power) / np.sum(N_power))
#     return snr_freq

# def real_fft_mag(ir, fs, db_scale=False, n_fft=None):
#     """Real FFT magnitude spectrum"""
#     if n_fft is None:
#         n_fft = len(ir)
#     mag = np.abs(rfft(ir, n=n_fft))
#     freq = rfftfreq(n_fft, d=1/fs)

#     if db_scale:
#         mag = np.clip(mag, 1e-12, None)  # Prevent log(0)
#         mag = 20 * np.log10(mag)

#     return freq, mag






def snr_band_limited(signal_vec, noise_vec, 
                     f_min: float, 
                     f_max: float, 
                     fs: float = 44100,
                     n_fft: int = 32768,   # now this is pwelch's nperseg
                     band_type: str = 'Bark',
                     f_ref: float = 1000):
    """Band limited SNR calculation using Welch PSD estimates."""

    # --- 1. Get band edges
    if band_type == "Bark":
        band_centers, band_edges, band_edge_matrix = get_bark_bands(f_min, f_max, fs)
    elif band_type in ["1-Octave", "1/3-Octave", "1/6-Octave"]:
        band_centers, band_edges, band_edge_matrix = get_octave_bands(
            f_min, f_max, fs, band_type=band_type, f_ref=f_ref)
    else:
        raise ValueError("Band type not defined. Use Bark, 1-Octave, 1/3-Octave, or 1/6-Octave")

    # --- 2. Welch PSD estimates
    f_signal, psd_signal = welch(signal_vec, fs=fs, nperseg=n_fft, scaling='density')
    f_noise, psd_noise = welch(noise_vec, fs=fs, nperseg=n_fft, scaling='density')
    # Note: 1- to calculate in true energy units, the psd should be multiplied by Δf, however wince I am only calculating SNR and they cancel out in the division no need for the moment 
    #       2- by default welch uses hann window


    # sanity check
    assert np.allclose(f_signal, f_noise), "Frequency axes mismatch in Welch PSD"
    freq = f_signal

    # --- 3. Band SNR computation
    snr_band = np.zeros((len(band_edge_matrix), 1))  # ensure (N,1)

    for idx, (f_lower, f_upper) in enumerate(band_edge_matrix):
        # Indices inside current band
        band_idx = np.where((freq >= f_lower) & (freq < f_upper))[0]

        if len(band_idx) == 0:
            print(f"Warning: n_fft={n_fft} → no bins in band {f_lower:.1f}-{f_upper:.1f} Hz. Setting SNR=0.")
            snr_band[idx, 0] = 0
            continue

        # Total band power = sum of PSD in band (approx integration)
        signal_power = np.sum(psd_signal[band_idx])
        noise_power = np.sum(psd_noise[band_idx])

        if noise_power <= 0:
            snr_band[idx, 0] = np.inf
        else:
            snr_band[idx, 0] = 10 * np.log10(signal_power / noise_power)

    return snr_band, band_centers, band_edges, band_edge_matrix







if __name__ == "__main__":
    # Example usage:
    # just run the file
    # my_signal = np.array([1, 2, 3, 4, 5])
    # noise_floor  = my_signal * 1e-8
    # snr_value = calculate_snr(my_signal)
    # print(f"SNR: {snr_value} dB")

    # Another example with a different noise floor
    # snr_value_low_noise = calculate_snr(my_signal, noise_floor)
    # print(f"SNR with lower noise: {snr_value_low_noise} dB")

    signal = 100*np.random.randn(512)
    noise = 0.1*np.random.randn(512)
    snr_vals, centers, edges, edge_matrix = snr_band_limited(
    signal, noise, f_min=100, f_max=20000, 
    fs=44100, n_fft=32768, band_type='Bark')

    # snrs = []
    # for _ in range(1000):  # Monte Carlo runs
    #     sig = 100*np.random.randn(512)
    #     noi = 0.1*np.random.randn(512)
    #     snr_band, centers, *_ = snr_band_limited(
    #         sig, noi, f_min=100, f_max=20000, fs=44100,
    #         n_fft=32768, band_type='Bark')
    #     snrs.append(snr_band.ravel())

    # mean_snr = np.mean(snrs, axis=0)


    # import matplotlib.pyplot as plt
    # plt.plot(centers, mean_snr)
    # plt.show()

    # Example with only HRIR (noise from tail)
    # snr_vals, centers, edges, edge_mat = snr_band_limited_advanced(
    #     signal, noise,
    #     f_min=100, f_max=20000, fs=44100,
    #     band_type='Bark',
    #     method='filterbank_hrir',
    #     fb_order=4,
    #     # optional: override windows if you already segmented the tail
    #     # signal_window=(s0, s1),
    #     # noise_window=(n0, n1),
    #     smooth_bark=True, smooth_sigma_bark=0.4
    # )
    import matplotlib.pyplot as plt
    plt.plot(centers, snr_vals)
    plt.show()

    

