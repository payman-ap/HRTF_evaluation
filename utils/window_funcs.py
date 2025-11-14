import numpy as np
from scipy.signal.windows import hann, hamming, kaiser

def window_gen(
    type,
    ramp_up_samples,
    plateau_samples,
    ramp_down_samples,
    symmetry_type=False,
    normalize_energy=False,
    beta=14  # Only used for Kaiser window
):
    # Normalize window type
    type = type.lower()
    type_aliases = {
        'hann': 'hann', 'hanning': 'hann',
        'hamm': 'hamming', 'hamming': 'hamming',
        'rect': 'rect', 'rectangular': 'rect',
        'kaiser': 'kaiser'
    }

    if type not in type_aliases:
        raise ValueError(f"Unsupported window type: {type}")
    
    win_type = type_aliases[type]

    def generate_section(kind, n):
        if n <= 0:
            return np.array([], dtype=float)
        if kind == 'hann':
            return hann(n * 2, sym=symmetry_type)[:n]
        elif kind == 'hamming':
            return hamming(n * 2, sym=symmetry_type)[:n]
        elif kind == 'kaiser':
            return kaiser(n * 2, beta=beta, sym=symmetry_type)[:n]
        elif kind == 'rect':
            return np.ones(n)
        else:
            raise ValueError(f"Unknown window kind: {kind}")

    def generate_ramp_down(kind, n):
        if n <= 0:
            return np.array([], dtype=float)
        if kind == 'hann':
            return hann(n * 2, sym=symmetry_type)[-n:]
        elif kind == 'hamming':
            return hamming(n * 2, sym=symmetry_type)[-n:]
        elif kind == 'kaiser':
            return kaiser(n * 2, beta=beta, sym=symmetry_type)[-n:]
        elif kind == 'rect':
            return np.ones(n)
        else:
            raise ValueError(f"Unknown window kind: {kind}")

    ramp_up = generate_section(win_type, ramp_up_samples)
    plateau = np.ones(plateau_samples) if plateau_samples > 0 else np.array([], dtype=float)
    ramp_down = generate_ramp_down(win_type, ramp_down_samples)

    window = np.concatenate([ramp_up, plateau, ramp_down])

    # Energy normalization
    # When to use: 
    #               Power Spectral Density (PSD) Estimation: Ensures the FFT-based power estimate is unbiased.
    #               Filter design: Maintains consistent gain in FIR filters when windows are applied.
    #               Comparing Different Windows: Allows fair comparison of leakage effects since all windows have the same energy.
    if normalize_energy and np.any(window):
        window /= np.sqrt(np.sum(window**2))

    return window

# Why symmetry type: FIR filters require linear phase (constant group delay), which is achieved only if the window is symmetric.
#   FIR Design: Use sym=True (default) to preserve linear phase.
#   FFT Analysis: Use sym=False to avoid a discontinuity at the edges.



def gaussmod(sig, risetime_ms=20, fs=44100):
    """
    Apply Gaussian envelope modulation to both ends of a signal.
    
    Parameters:
        sig (np.ndarray): Input signal (1D or 2D, with shape [samples, channels]).
        risetime_ms (float): Time in ms from 10% to 90% amplitude (default 20 ms).
        fs (float): Sampling rate in Hz (default 44100).
    
    Returns:
        y (np.ndarray): Signal after applying the Gaussian envelope.
        env (np.ndarray): Envelope used for modulation (1D).
    """
    sig = np.atleast_2d(sig)
    if sig.shape[0] < sig.shape[1]:
        sig = sig.T

    len_sig = sig.shape[0]

    if risetime_ms > 0:
        k = (np.sqrt(np.log(1 / 0.9)) - np.sqrt(np.log(1 / 0.1))) ** 2 / risetime_ms ** 2
        delta = 10 ** (-60 / 20)
        tc = np.sqrt(-np.log(delta) / k)
        t_e = np.arange(-tc, tc + 1 / (fs / 1000), 1 / (fs / 1000))  # in ms

        if len(t_e) > len_sig:
            raise ValueError("Signal must be longer than twice the Gaussian risetime.")

        ye = np.exp(-k * t_e ** 2)
        i = np.argmax(ye)

        # Scalar expansion to match signal shape
        Ein = ye[:i+1][:, np.newaxis]
        Aus = ye[i:][:, np.newaxis]

        # Pad to match shape
        Ein = np.tile(Ein, (1, sig.shape[1]))
        Aus = np.tile(Aus, (1, sig.shape[1]))

        # Apply envelope
        sig[:i+1, :] *= Ein
        sig[-len(Aus):, :] *= Aus

        env = np.concatenate([
            Ein[:, 0],
            np.ones(len_sig - (len(Ein) + len(Aus))),
            Aus[:, 0]
        ])
    else:
        env = np.ones(len_sig)

    return sig.squeeze(), env










if __name__ == "__main__":
    import matplotlib.pyplot as plt

    w0 = window_gen('hann', 50, 412, 50, symmetry_type=True)
    w1 = window_gen('hann', 50, 412, 50, symmetry_type=False)
    w2 = window_gen('hamming', 50, 412, 50)
    w3 = window_gen('rect', 50, 412, 50)
    w4 = window_gen('hann', 50, 412, 50, normalize_energy=True)
    print(w0.shape, w1.shape, w2.shape, w3.shape)  # (40,)
    plt.plot(w1)
    plt.plot(w2)
    plt.plot(w3)
    plt.plot(w4)
    plt.show()

    sig = np.ones((512, 1))
    risetime_ms = 40/44100*1000
    _, win = gaussmod(sig, risetime_ms=risetime_ms, fs=44100)
    plt.plot(win)
    plt.show()
