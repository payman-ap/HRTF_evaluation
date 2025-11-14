import numpy as np
from scipy.signal import cheby1, cheb1ord, firwin, lfilter
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt

def hz2bark(f, method='traunc'):
    return 7 * np.arcsinh(f / 650)

def bark2hz(z, method='traunc'):
    return 650 * np.sinh(z / 7)

def bp_filter(sig, flow, fup, method='fft', fs=44100):
    Fn = fs / 2
    # If we'd like bp_filter to always work seamlessly on both 1D and 2D signals (across all methods)
    sig = np.atleast_2d(sig)
    if sig.shape[0] < sig.shape[1]:  # ensure it's (samples, channels)
        sig = sig.T
    ################


    if method == 'masking':
        Wpl = fup / Fn
        Wph = flow / Fn
        Rs = 60
        Rp = 1

        zo = 60 / 27 + hz2bark(fup)
        Wsl = bark2hz(zo) / Fn
        Wsl = np.clip(Wsl, np.finfo(float).eps, 0.99999999)

        zu = hz2bark(flow) - 60 / 27
        Wsh = bark2hz(zu) / Fn
        Wsh = np.clip(Wsh, np.finfo(float).eps, 1)

        n, Wn = cheb1ord(Wpl, Wsl, Rp, Rs)
        bl, al = cheby1(n, Rp, Wn)
        sigtp = lfilter(bl, al, sig, axis=0)

        n, Wn = cheb1ord(Wph, Wsh, Rp, Rs)
        bh, ah = cheby1(n, Rp, Wn, btype='high')
        y = lfilter(bh, ah, sigtp, axis=0)

    elif method == 'fft':
        if sig.ndim == 1:
            sig = sig[:, np.newaxis]
        sig_len = sig.shape[0]

        y = np.zeros_like(sig)
        fft_idx_low = round(sig_len / fs * flow)
        fft_idx_high = round(sig_len / fs * fup)

        for i in range(sig.shape[1]):
            sig_fft = fft(sig[:, i], sig_len)
            sig_fft_bp_pos = np.zeros(sig_len // 2 + 1, dtype=complex)
            sig_fft_bp_pos[fft_idx_low:fft_idx_high] = sig_fft[fft_idx_low:fft_idx_high]

            sig_fft_bp_cmplx = np.concatenate(
                [sig_fft_bp_pos, np.conj(sig_fft_bp_pos[-2:0:-1])]
            )
            y[:, i] = np.real(ifft(sig_fft_bp_cmplx, sig_len))

    elif method in ['fir256', 'fir512', 'fir1024']:
        if sig.ndim == 1:
            sig = sig[:, np.newaxis]

        taps = int(method[3:])
        B = firwin(taps, [flow / Fn, fup / Fn], pass_zero=False)
        y = lfilter(B, 1, sig, axis=0)

    return y







if __name__ == '__main__':
    # Example code for the functionality test
    fs = 44100
    t = np.linspace(0, 1, fs, endpoint=False)
    sig = np.sin(2 * np.pi * 1000 * t) + np.sin(2 * np.pi * 5000 * t)
    sig = sig[:, np.newaxis]

    flow = 800
    fup = 1200
    methods = ['masking', 'fft', 'fir256', 'fir512', 'fir1024']
    filtered_signals = {method: bp_filter(sig, flow, fup, method, fs) for method in methods}

    # Time-domain plots
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 2, 1)
    plt.plot(t, sig)
    plt.title('Original Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')

    for i, method in enumerate(methods, 2):
        plt.subplot(3, 2, i)
        plt.plot(t, filtered_signals[method])
        plt.title(f'Filtered Signal ({method})')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()

    # Frequency-domain plots
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 2, 1)
    freqs = np.fft.fftfreq(sig.shape[0], 1/fs)
    plt.plot(freqs[:sig.shape[0] // 2], 20 * np.log10(np.abs(fft(sig[:, 0]))[:sig.shape[0] // 2]))
    plt.title('Original Signal Spectrum')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [dB]')

    for i, method in enumerate(methods, 2):
        plt.subplot(3, 2, i)
        plt.plot(freqs[:sig.shape[0] // 2], 20 * np.log10(np.abs(fft(filtered_signals[method][:, 0]))[:sig.shape[0] // 2]))
        plt.title(f'Filtered Signal Spectrum ({method})')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Magnitude [dB]')

    plt.tight_layout()
    plt.show()




