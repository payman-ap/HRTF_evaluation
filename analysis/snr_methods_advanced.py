
import numpy as np
from scipy.signal import welch, butter, sosfiltfilt, periodogram
from scipy.signal.windows import dpss
from numpy.fft import rfft, rfftfreq

# for importing from hrtf_eval_py package
import sys
import os
# Go up 2 levels: from /hrtf_eval/notebooks to /my_project
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from hrtf_eval_py.utils.get_frequency_bands import get_bark_bands, get_octave_bands

# --------- Helpers ---------

def _integrate_band_power(freq, psd, f_lo, f_hi):
    """Integrate PSD (power/Hz) over [f_lo, f_hi] using trapezoidal rule."""
    idx = np.where((freq >= f_lo) & (freq < f_hi))[0]
    if idx.size == 0:
        return 0.0, False
    return float(np.trapz(psd[idx], freq[idx])), True

def _design_band_filter(fs, f_lo, f_hi, order=4):
    """
    Stable zero-phase IIR design per band using Butterworth in SOS form.
    Handles lowpass / highpass cases for edge bands.
    """
    nyq = fs * 0.5
    eps = 1e-9
    f_lo = max(f_lo, 0.0)
    f_hi = min(f_hi, nyq - 1e-6)

    if f_lo <= eps:  # lowpass
        wn = f_hi / nyq
        wn = np.clip(wn, 1e-6, 0.999999)
        sos = butter(order, wn, btype='low', output='sos')
    elif f_hi >= nyq - eps:  # highpass
        wn = f_lo / nyq
        wn = np.clip(wn, 1e-6, 0.999999)
        sos = butter(order, wn, btype='high', output='sos')
    else:  # bandpass
        wn = [f_lo / nyq, f_hi / nyq]
        wn[0] = np.clip(wn[0], 1e-6, 0.999)
        wn[1] = np.clip(wn[1], wn[0] + 1e-6, 0.999999)
        sos = butter(order, wn, btype='band', output='sos')
    return sos

def _band_energy_filterbank(x, fs, band_edge_matrix, filter_order=4):
    """
    For each band, zero-phase filter and compute sum of squares (energy).
    Returns energies as (Nbands,) ndarray.
    """
    energies = np.zeros(len(band_edge_matrix), dtype=float)
    for i, (f_lo, f_hi) in enumerate(band_edge_matrix):
        sos = _design_band_filter(fs, f_lo, f_hi, order=filter_order)
        # zero-phase filtering (good for HRIRs; preserves timing/peaks)
        y = sosfiltfilt(sos, x)
        energies[i] = np.sum(y * y)
    return energies

def _psd_welch(x, fs, nfft=32768, nperseg=None, noverlap=None, window='hann'):
    if nperseg is None:
        nperseg = min(len(x), 512)  # safe default for 512-sample HRIRs
    if noverlap is None:
        noverlap = nperseg // 2
    f, Pxx = welch(x, fs=fs, window=window, nperseg=nperseg,
                   noverlap=noverlap, nfft=nfft, scaling='density', detrend=False)
    return f, Pxx

def _psd_multitaper(x, fs, nfft=32768, NW=2.5, Kmax=None, adaptive=False):
    """
    Multitaper PSD via averaging DPSS-tapered periodograms.
    Uses scipy.signal.periodogram with each taper as the window.
    """
    N = len(x)
    if Kmax is None:
        Kmax = max(1, int(2 * NW - 1))  # common choice
    tapers = dpss(N, NW=NW, Kmax=Kmax, sym=False)
    Pxx_accum = None
    f_ref = None
    for k in range(tapers.shape[0]):
        f, Pk = periodogram(x, fs=fs, window=tapers[k], nfft=nfft,
                            scaling='density', detrend=False, return_onesided=True)
        if Pxx_accum is None:
            Pxx_accum = np.array(Pk, dtype=float)
            f_ref = f
        else:
            Pxx_accum += Pk
    Pxx_mt = Pxx_accum / tapers.shape[0]
    return f_ref, Pxx_mt


def snr_band_limited(signal_vec, noise_vec, 
                     f_min: float, 
                     f_max: float, 
                     fs: float = 44100,
                     band_type: str = 'Bark',
                     f_ref: float = 1000.0,
                     method: str = 'filterbank',
                     # Welch params
                     nfft: int = 32768,
                     nperseg: int = None,
                     noverlap: int = None,
                     window: str = 'hann',
                     # Multitaper params
                     mt_NW: float = 2.5,
                     mt_Kmax: int = None,
                     # Filterbank params
                     fb_order: int = 4,
                     # Numerics
                     noise_floor_ratio: float = 1e-12):
    """
    Band-limited SNR across Bark/Octave bands with multiple methods:
      - method='filterbank'  : zero-phase Butterworth per band → energy ratio (stable for short HRIRs)
      - method='welch'       : integrate Welch PSD over each band (trapz)
      - method='multitaper'  : integrate DPSS-multitaper PSD over each band (trapz)

    Returns:
      snr_band         : (Nbands, 1) column vector, SNR (dB)
      band_centers     : (Nbands,)     band center freqs
      band_edges       : (Nbands+1,)   edges
      band_edge_matrix : (Nbands, 2)   [[f_lo, f_hi], ...]
    """

    # 1) Bands
    if band_type == "Bark":
        band_centers, band_edges, band_edge_matrix = get_bark_bands(f_min, f_max, fs)
    elif band_type in ["1-Octave", "1/3-Octave", "1/6-Octave"]:
        band_centers, band_edges, band_edge_matrix = get_octave_bands(
            f_min, f_max, fs, band_type=band_type, f_ref=f_ref)
    else:
        raise ValueError("Band type not defined. Use 'Bark', '1-Octave', '1/3-Octave', or '1/6-Octave'.")

    band_edge_matrix = np.asarray(band_edge_matrix, dtype=float)

    # 2) Method-specific power per band
    if method.lower() == 'filterbank':
        # Time-domain energy in each band via zero-phase filtering
        sig_E = _band_energy_filterbank(signal_vec, fs, band_edge_matrix, filter_order=fb_order)
        noi_E = _band_energy_filterbank(noise_vec,  fs, band_edge_matrix, filter_order=fb_order)

        # 3) SNR from energies
        # Regularize tiny denominators to avoid blow-ups
        total_noi_E = float(np.sum(noi_E))
        eps = max(noise_floor_ratio * total_noi_E, np.finfo(float).tiny)
        snr_lin = sig_E / np.maximum(noi_E, eps)
        snr_db = 10.0 * np.log10(np.maximum(snr_lin, np.finfo(float).tiny))

    elif method.lower() == 'welch':
        freq_s, Pxx_s = _psd_welch(signal_vec, fs, nfft=nfft, nperseg=nperseg, noverlap=noverlap, window=window)
        freq_n, Pxx_n = _psd_welch(noise_vec,  fs, nfft=nfft, nperseg=nperseg, noverlap=noverlap, window=window)
        if not np.allclose(freq_s, freq_n):
            raise RuntimeError("Welch frequency axes mismatch.")
        snr_vals = []
        # Precompute full-band noise for floor
        full_pow_n, _ = _integrate_band_power(freq_n, Pxx_n, f_min, f_max)
        eps = max(noise_floor_ratio * full_pow_n, np.finfo(float).tiny)
        for f_lo, f_hi in band_edge_matrix:
            sig_pow, has_bins_s = _integrate_band_power(freq_s, Pxx_s, f_lo, f_hi)
            noi_pow, has_bins_n = _integrate_band_power(freq_n, Pxx_n, f_lo, f_hi)
            if not (has_bins_s and has_bins_n):
                # With large nfft this is rare; keep consistent output with a warning-like behavior.
                snr_vals.append(0.0)
            else:
                snr_vals.append(10.0 * np.log10(max(sig_pow, 0.0) / max(noi_pow, eps)))
        snr_db = np.asarray(snr_vals, dtype=float)

    elif method.lower() == 'multitaper':
        freq_s, Pxx_s = _psd_multitaper(signal_vec, fs, nfft=nfft, NW=mt_NW, Kmax=mt_Kmax)
        freq_n, Pxx_n = _psd_multitaper(noise_vec,  fs, nfft=nfft, NW=mt_NW, Kmax=mt_Kmax)
        if not np.allclose(freq_s, freq_n):
            raise RuntimeError("Multitaper frequency axes mismatch.")
        snr_vals = []
        full_pow_n, _ = _integrate_band_power(freq_n, Pxx_n, f_min, f_max)
        eps = max(noise_floor_ratio * full_pow_n, np.finfo(float).tiny)
        for f_lo, f_hi in band_edge_matrix:
            sig_pow, has_bins_s = _integrate_band_power(freq_s, Pxx_s, f_lo, f_hi)
            noi_pow, has_bins_n = _integrate_band_power(freq_n, Pxx_n, f_lo, f_hi)
            if not (has_bins_s and has_bins_n):
                snr_vals.append(0.0)
            else:
                snr_vals.append(10.0 * np.log10(max(sig_pow, 0.0) / max(noi_pow, eps)))
        snr_db = np.asarray(snr_vals, dtype=float)

    else:
        raise ValueError("Unknown method. Choose from: 'filterbank', 'welch', 'multitaper'.")

    # 4) Ensure column-vector output (N,1)
    snr_band = np.asarray(snr_db, dtype=float).reshape((-1, 1))
    return snr_band, band_centers, band_edges, band_edge_matrix









def snr_band_limited_old(signal_vec, noise_vec, 
                     f_min: float, 
                     f_max: float, 
                     fs: float = 44100,
                     nfft: int = 32768,        # zero-padded FFT size (like MATLAB pwelch's 3rd arg)
                     nperseg: int = None,      # Welch window length (≤ len(signal)), default set below
                     noverlap: int = None,     # Welch overlap; default 50%
                     window: str = 'hann',
                     band_type: str = 'Bark',
                     f_ref: float = 1000.0,
                     noise_floor_ratio: float = 1e-12):
    """
    Band-limited SNR using Welch PSD with proper integration over frequency.
    Returns:
        snr_band: (Nbands, 1) column vector of SNRs in dB
        band_centers, band_edges, band_edge_matrix
    """

    # 1) Bark / Octave bands
    if band_type == "Bark":
        band_centers, band_edges, band_edge_matrix = get_bark_bands(f_min, f_max, fs)
    elif band_type in ["1-Octave", "1/3-Octave", "1/6-Octave"]:
        band_centers, band_edges, band_edge_matrix = get_octave_bands(
            f_min, f_max, fs, band_type=band_type, f_ref=f_ref)
    else:
        raise ValueError("Band type not defined. Use 'Bark', '1-Octave', '1/3-Octave', or '1/6-Octave'.")

    # 2) Welch PSD settings
    if nperseg is None:
        nperseg = min(len(signal_vec), 512)  # safe default for your 512-sample signals
    if noverlap is None:
        noverlap = nperseg // 2

    # Welch PSD (density = power/Hz)
    f_sig, psd_signal = welch(signal_vec, fs=fs, window=window,
                              nperseg=nperseg, noverlap=noverlap,
                              nfft=nfft, scaling='density', detrend=False)
    f_noi, psd_noise  = welch(noise_vec,  fs=fs, window=window,
                              nperseg=nperseg, noverlap=noverlap,
                              nfft=nfft, scaling='density', detrend=False)

    if not np.allclose(f_sig, f_noi):
        raise RuntimeError("Welch frequency axes mismatch.")

    freq = f_sig

    # 3) Integrate PSD over each band (trapz), compute SNR
    snr_band = np.zeros((len(band_edge_matrix), 1))
    # Precompute a global small floor (optional) based on total noise power across [f_min, f_max]
    full_idx = np.where((freq >= f_min) & (freq <= f_max))[0]
    total_noise_power = float(np.trapz(psd_noise[full_idx], freq[full_idx])) if len(full_idx) else 0.0
    eps_floor = max(noise_floor_ratio * total_noise_power, np.finfo(float).tiny)

    for i, (f_lo, f_hi) in enumerate(band_edge_matrix):
        band_idx = np.where((freq >= f_lo) & (freq < f_hi))[0]

        if len(band_idx) == 0:
            print(f"Warning: nfft={nfft} resolution gave no bins in {f_lo:.2f}–{f_hi:.2f} Hz. SNR=0.")
            snr_band[i, 0] = 0.0
            continue

        # Proper integration (power = ∫ PSD df)
        sig_pow = float(np.trapz(psd_signal[band_idx], freq[band_idx]))
        noi_pow = float(np.trapz(psd_noise[band_idx],  freq[band_idx]))

        noi_pow = max(noi_pow, eps_floor)  # regularize denominator
        snr_band[i, 0] = 10.0 * np.log10(sig_pow / noi_pow)

    return snr_band, band_centers, band_edges, band_edge_matrix








# Second implementation of the advanced filtered approach
from scipy.signal import butter, sosfiltfilt

def _design_band_filter(fs, f_lo, f_hi, order=4):
    nyq = fs * 0.5
    f_lo = max(0.0, f_lo)
    f_hi = min(f_hi, nyq - 1e-6)
    if f_lo <= 1e-9:  # lowpass
        wn = np.clip(f_hi / nyq, 1e-6, 0.999999)
        sos = butter(order, wn, btype='low', output='sos')
    elif f_hi >= nyq - 1e-9:  # highpass
        wn = np.clip(f_lo / nyq, 1e-6, 0.999999)
        sos = butter(order, wn, btype='high', output='sos')
    else:  # bandpass
        w = [f_lo / nyq, f_hi / nyq]
        w[0] = np.clip(w[0], 1e-6, 0.999)
        w[1] = np.clip(w[1], w[0] + 1e-6, 0.999999)
        sos = butter(order, w, btype='band', output='sos')
    return sos

def _bark_smoothing(snr_db, centers_bark, sigma_bark=0.4):
    """Small Gaussian blur across Bark bands to stabilize tiny ripples."""
    z = centers_bark  # already on Bark axis if you pass Bark centers; otherwise map first
    snr_db = snr_db.astype(float).ravel()
    W = np.exp(-0.5*((z[:,None]-z[None,:])/max(sigma_bark,1e-6))**2)
    W /= np.maximum(W.sum(axis=1, keepdims=True), np.finfo(float).tiny)
    return (W @ snr_db).reshape((-1,1))

def _auto_windows(hrir, fs, peak_pad_ms=0.2, sig_ms=8.0, noise_ms=20.0, noise_gap_ms=1.0):
    """
    Build default windows from a single HRIR:
      - find main peak
      - start signal a tiny bit before it (peak_pad_ms)
      - signal window duration sig_ms
      - noise window starts noise_gap_ms after end of signal, with duration noise_ms
    """
    N = len(hrir)
    peak = int(np.argmax(np.abs(hrir)))
    def ms2samp(ms): return int(round(ms * 1e-3 * fs))

    s0 = max(0, peak - ms2samp(peak_pad_ms))
    s1 = min(N, s0 + ms2samp(sig_ms))
    n0 = min(N, s1 + ms2samp(noise_gap_ms))
    n1 = min(N, n0 + ms2samp(noise_ms))

    # If tail too short, fallback to last part of IR
    if n1 - n0 < ms2samp(5.0):  # need at least 5 ms for a stable noise window
        n1 = N
        n0 = max(s1 + ms2samp(noise_gap_ms), N - ms2samp(10.0))
    return (s0, s1), (n0, n1)

def snr_band_limited_advanced2(signal_vec, noise_vec=None,
                     f_min: float = 100.0,
                     f_max: float = 20000.0,
                     fs: float = 44100.0,
                     band_type: str = 'Bark',
                     f_ref: float = 1000.0,
                     method: str = 'filterbank_hrir',
                     # HRIR method params
                     fb_order: int = 4,
                     signal_window: tuple | None = None,  # (start_idx, end_idx)
                     noise_window: tuple | None = None,   # (start_idx, end_idx)
                     peak_pad_ms: float = 0.2,
                     sig_ms: float = 8.0,
                     noise_ms: float = 20.0,
                     noise_gap_ms: float = 1.0,
                     # smoothing
                     smooth_bark: bool = True,
                     smooth_sigma_bark: float = 0.4,
                     # numeric floor
                     noise_floor_ratio: float = 1e-12):
    """
    Band-limited SNR for short HRIRs from a *single* recording using noise-bias-corrected
    per-band energies. If `noise_vec` is None, uses the HRIR tail as noise reference.

    Returns:
      snr_band (N,1), band_centers, band_edges, band_edge_matrix
    """

    # 0) Bands
    if band_type == "Bark":
        band_centers, band_edges, band_edge_matrix = get_bark_bands(f_min, f_max, fs)
        centers_for_smoothing = band_centers  # assume Bark units
    elif band_type in ["1-Octave", "1/3-Octave", "1/6-Octave"]:
        band_centers, band_edges, band_edge_matrix = get_octave_bands(
            f_min, f_max, fs, band_type=band_type, f_ref=f_ref)
        centers_for_smoothing = band_centers  # on Hz; smoothing will still help slightly
    else:
        raise ValueError("Unsupported band_type.")

    band_edge_matrix = np.asarray(band_edge_matrix, float)

    if method.lower() != 'filterbank_hrir':
        raise ValueError("For single HRIR stability, use method='filterbank_hrir' here.")

    x = np.asarray(signal_vec, float)

    # 1) Build windows
    if noise_vec is not None:
        # Explicit noise track provided: use full length for noise window unless overridden
        if signal_window is None or noise_window is None:
            (s0, s1), (n0, n1) = _auto_windows(x, fs, peak_pad_ms, sig_ms, noise_ms, noise_gap_ms)
            if signal_window is None: signal_window = (s0, s1)
            if noise_window  is None: noise_window  = (0, len(noise_vec))  # whole vector as noise
    else:
        # Single HRIR only: use its *tail* as noise reference
        if signal_window is None or noise_window is None:
            signal_window, noise_window = _auto_windows(x, fs, peak_pad_ms, sig_ms, noise_ms, noise_gap_ms)
        noise_vec = x  # measure noise from HRIR tail

    s0, s1 = signal_window
    n0, n1 = noise_window
    s0 = max(0, min(len(x), s0)); s1 = max(s0+1, min(len(x), s1))
    n0 = max(0, min(len(noise_vec), n0)); n1 = max(n0+1, min(len(noise_vec), n1))

    # 2) Per-band zero-phase filtering and energy computation with bias correction
    sig_E = np.zeros(len(band_edge_matrix), float)
    noi_E = np.zeros(len(band_edge_matrix), float)

    lenS = s1 - s0
    lenN = n1 - n0
    if lenN < int(0.005 * fs):  # <5 ms
        print("Noise window very short; consider increasing noise_ms or moving noise_window.")
    # extract windows
    x_sig = x[s0:s1]
    x_noi = np.asarray(noise_vec, float)[n0:n1]

    # tiny fade windows to reduce leakage at cuts
    def cosine_fade(N):
        if N <= 4: return np.ones(N)
        n = np.arange(N)
        win = np.ones(N)
        L = max(1, N//50)  # ~2% fade
        ramp = 0.5*(1 - np.cos(np.pi*np.arange(L)/L))
        win[:L] = ramp
        win[-L:] = ramp[::-1]
        return win

    w_sig = cosine_fade(lenS)
    w_noi = cosine_fade(lenN)

    for i, (f_lo, f_hi) in enumerate(band_edge_matrix):
        sos = _design_band_filter(fs, f_lo, f_hi, order=fb_order)

        y_sig = sosfiltfilt(sos, x_sig) * w_sig
        y_noi = sosfiltfilt(sos, x_noi) * w_noi

        E_sig = float(np.sum(y_sig*y_sig))
        E_noi = float(np.sum(y_noi*y_noi))

        # noise-bias correction inside signal window
        E_sig_corr = max(E_sig - (lenS/lenN) * E_noi, 0.0)

        sig_E[i] = E_sig_corr
        noi_E[i] = E_noi

    # 3) SNR with floors + (optional) Bark smoothing
    total_noi = float(np.sum(noi_E))
    eps = max(noise_floor_ratio * total_noi, np.finfo(float).tiny)
    snr_lin = sig_E / np.maximum(noi_E, eps)
    snr_db = 10.0 * np.log10(np.maximum(snr_lin, np.finfo(float).tiny))
    snr_band = snr_db.reshape((-1,1))

    if smooth_bark and len(band_edge_matrix) >= 3:
        snr_band = _bark_smoothing(snr_band, np.asarray(centers_for_smoothing, float),
                                   sigma_bark=smooth_sigma_bark)

    return snr_band, band_centers, band_edges, band_edge_matrix








# Testing functions above
if __name__ == "__main__":
    # Example usage:
    # just run the file

    signal = 100*np.random.randn(512)
    noise = 0.1*np.random.randn(512)

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
    snr_vals, centers, edges, edge_mat = snr_band_limited_advanced2(
        signal, noise,
        f_min=100, f_max=20000, fs=44100,
        band_type='Bark',
        method='filterbank_hrir',
        fb_order=4,
        # optional: override windows if you already segmented the tail
        # signal_window=(s0, s1),
        # noise_window=(n0, n1),
        smooth_bark=True, smooth_sigma_bark=0.4
    )
    import matplotlib.pyplot as plt
    plt.plot(centers, snr_vals)
    plt.show()

    

