import numpy as np
from scipy.fft import rfft, rfftfreq

def real_fft_mag(ir, fs, db_scale=False, n_fft=None):
    if n_fft is None:
        n_fft = len(ir)
    mag = np.abs(rfft(ir, n=n_fft))
    freq = rfftfreq(n_fft, d=1/fs)
    
    if db_scale:
        mag = np.clip(mag, 1e-12, None)  # Prevent log(0)
        mag = 20 * np.log10(mag)
        
    return freq, mag

def real_fft_phase(ir, fs, unwrap=False, n_fft=None):
    if n_fft is None:
        n_fft = len(ir)
    
    fft_result = rfft(ir, n=n_fft)
    phase = np.angle(fft_result)
    
    if unwrap:
        phase = np.unwrap(phase)
        
    freq = rfftfreq(n_fft, d=1/fs)
    return freq, phase







def compute_magnitude_spectrum_db(signal, fs):
    """
    Computes one-sided magnitude spectrum in dB.
    Returns frequency and dB-magnitude arrays.
    """
    N = len(signal)
    fft_result = np.fft.fft(signal)
    magnitude = np.abs(fft_result)[:N // 2]
    magnitude_db = 20 * np.log10(magnitude + 1e-12)  # avoid log(0)
    freqs = np.fft.fftfreq(N, d=1/fs)[:N // 2]
    return freqs, magnitude_db