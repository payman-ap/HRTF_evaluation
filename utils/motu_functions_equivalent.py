from scipy.signal import cheby1, cheb1ord, filtfilt
import numpy as np

import warnings



import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..'))) 
from hrtf_eval_py.utils.window_funcs import gaussmod
from hrtf_eval_py.utils.bp_filter import bp_filter



def motu_bandlim(sig, f_lo=300, f_up=18000, Fs=44100):
    """
    Bandpass filters the signal between f_lo and f_up using Chebyshev Type I filter.
    
    Parameters:
        sig (np.ndarray): Input signal.
        f_lo (float): Lower cutoff frequency in Hz.
        f_up (float): Upper cutoff frequency in Hz.
        Fs (int): Sampling frequency in Hz.
    
    Returns:
        np.ndarray: Filtered signal.
    """
    Fn = Fs / 2
    Wp_lo = f_lo / Fn
    Ws_lo = 0.8 * Wp_lo
    Wp_up = f_up / Fn
    Ws_up = min(1.25 * Wp_up, 0.99)

    Rp = 0.2  # Passband ripple (dB)
    Rs = 20   # Stopband attenuation (dB)

    # Design Chebyshev Type I bandpass filter
    n, Wn = cheb1ord([Wp_lo, Wp_up], [Ws_lo, Ws_up], Rp, Rs)
    b, a = cheby1(n, Rp, Wn, btype='band')

    # Apply filter
    y = filtfilt(b, a, sig)
    
    # Retry with lower order if output is zero
    while np.max(np.abs(y)) == 0 and n > 2:
        n -= 2
        b, a = cheby1(n, Rp, Wn, btype='band')
        y = filtfilt(b, a, sig)

    return y


def motu_rms(sig, wide=1, fs=44100):
    """
    Calculate the RMS level in dB SPL and the effective pressure of a signal.
    
    Parameters:
        sig (np.ndarray): Input signal.
        wide (int): Frequency range mode (0 = 200-12kHz, 1 = wideband, 2 = 200-8kHz).
        fs (int): Sampling frequency in Hz.
    
    Returns:
        tuple: (L, peff) where L is the level in dB SPL and peff is the effective pressure.
    """
    pmedav = 1 / 3.2905
    lmotu = 90 - 20 * np.log10(pmedav)

    if wide == 0:
        warnings.warn("Using motu_rms on a range of 200Hz to 12kHz.")
        sig = motu_bandlim(sig, 200, 12000, fs)
    elif wide == 2:
        warnings.warn("Using motu_rms on a range of 200Hz to 8kHz.")
        sig = motu_bandlim(sig, 200, 8000, fs)

    peff = np.sqrt(np.mean(sig ** 2))
    if peff == 0:
        peff = np.finfo(float).tiny  # Avoid log(0)

    L = 20 * np.log10(peff) + lmotu
    return L, peff



import numpy as np

def sig_duration(t, t_rise, option='gauss'):
    """
    Computes overall duration of signal after modulation, matching MATLAB's sig_duration logic.
    
    t        : target duration in seconds
    t_rise   : rise time in milliseconds
    option   : modulation type ('gauss', 'cos0', 'cos1')
    """
    if option == 'gauss':
        if t_rise > 0:
            k = (np.sqrt(np.log(1 / 0.9)) - np.sqrt(np.log(1 / 0.1)))**2 / (t_rise**2)
            delta = 10 ** (-60 / 20)
            tc = np.sqrt(-np.log(delta) / k)
            t67 = np.sqrt(-np.log(0.675) / k)
            dur = t + 2 * (tc - t67) / 1000  # convert ms to sec
        else:
            dur = t

    elif option == 'cos0':
        t67 = np.arccos(0.35) / np.pi
        dur = t + 2 * (1 - t67) * t_rise / 1000

    elif option == 'cos1':
        t_rise_corr = np.pi / (np.pi - 2 * np.arccos(0.8)) * t_rise
        t67 = np.arccos(0.35) / np.pi
        dur = t + 2 * (1 - t67) * t_rise_corr / 1000

    else:
        raise ValueError("Unknown option in sig_duration")

    return dur


def motu_noise(level_db, f_lo, f_up, duration, t_rise=0, Fs=44100):
    """
    Generate spectrally-shaped Gaussian noise using the specified parameters.
    
    level_db : desired output level in dB SPL (wideband after filtering)
    f_lo     : lower cutoff frequency in Hz
    f_up     : upper cutoff frequency in Hz
    duration : duration of signal at 67.5% envelope points (in seconds)
    t_rise   : optional Gaussian envelope rise time in milliseconds
    Fs       : sampling frequency
    """

    if isinstance(t_rise, str):
        t_rise = 20  # old 'g' option default

    if f_up > 20000 or f_lo < 20:
        print("Warning: Frequencies out of typical range. Headphone usage assumed.")

    # Adjust duration based on rise time
    t_total = sig_duration(duration, t_rise)

    # Generate raw white noise
    n_samples = int(np.round(t_total * Fs))
    noise = np.random.randn(n_samples)

    # Apply bandpass filter (assumed implemented)
    noise_bp = bp_filter(noise, f_lo, f_up, method='masking', fs=Fs)

    # Normalize to specified level
    rms_val, _ = motu_rms(noise_bp, wide=1)

    scale_factor = 10 ** ((level_db - rms_val) / 20)
    y = scale_factor * noise_bp

    # Apply Gaussian envelope if t_rise > 0
    if t_rise > 0:
        y = gaussmod(y, t_rise, Fs=Fs)

    return y





if __name__=='__main__':
    levelCalibNoise = motu_noise(70, 200, 20000, 5, t_rise=0, Fs=44100)
    rms_val, _ = motu_rms(levelCalibNoise, wide=1)
    print('noise_level:', rms_val)





