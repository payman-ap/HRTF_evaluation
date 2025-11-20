import numpy as np
from scipy.signal import chirp

def sweep_generate_exponential(
    sweep_duration,
    silence_duration,
    sample_rate,
    f_start,
    f_end,
    fade_in_time=0.08,
    fade_out_time=0.005,
):
    """
    Pure NumPy implementation of an exponential sine sweep (Farina/Schroeder)
    and its matching inverse filter.

    Returns
    -------
    sweep : np.ndarray
        The sweep including fades and silence.
    inverse : np.ndarray
        The matched inverse filter padded to the same total length.
    """

    if sweep_duration > 100:
        raise ValueError("Sweep duration > 100 s is not supported.")

    sr = sample_rate
    T = sweep_duration
    t = np.arange(0, T, 1/sr)

    # ================================================
    # 1) True exponential sweep formula (Farina)
    # ================================================
    #
    #   sweep(t) = sin( 2π * f1 * T / ln(f2/f1) * ( exp( t/T * ln(f2/f1) ) - 1 ) )
    #
    K = T * f_start / np.log(f_end / f_start)
    L = np.log(f_end / f_start)

    phase = 2 * np.pi * K * (np.exp(t * L / T) - 1)
    sweep = np.sin(phase)

    # ================================================
    # 2) Fade-in and fade-out
    # ================================================
    fade_in_len = int(np.ceil(fade_in_time * sr))
    fade_out_len = int(np.ceil(fade_out_time * sr))

    if fade_in_len > 0:
        fade_in = np.sin(0.5 * np.pi * np.linspace(0, 1, fade_in_len, endpoint=False))
        sweep[:fade_in_len] *= fade_in

    if fade_out_len > 0:
        fade_out = np.sin(0.5 * np.pi * np.linspace(1, 0, fade_out_len, endpoint=False))
        sweep[-fade_out_len:] *= fade_out

    # ================================================
    # 3) Trailing silence
    # ================================================
    end_silence = np.zeros(int(np.ceil(silence_duration * sr)))
    sweep_with_silence = np.concatenate([sweep, end_silence])

    # ================================================
    # 4) Compute inverse filter (Schroeder inverse)
    # ================================================
    #
    # Matching envelope:
    #   env(t) = exp( t * ln(f2/f1) / T )
    #
    # Inverse filter:
    #   inverse(t) = sweep_flipped(t) / env(t)
    # ================================================

    env = np.exp(t * L / T)

    # Flip the sweep *before* the silence
    sweep_flipped = sweep[::-1]

    inverse = sweep_flipped / env

    # Pad to match length of sweep+silence
    inverse_padded = np.concatenate([inverse, np.zeros_like(end_silence)])

    return sweep_with_silence, inverse_padded


def simple_ess(f1:int, f2:int, t:int, T:float) -> float:
    # make sweep
    f_t = f1 * np.exp( (t/T) * np.log(f2/f1) )
    L = np.log(f2/f1)
    phase_t = 2*np.pi*(T*f1/L)*( np.exp(t*L/T)-1 )
    s_t = np.sin(phase_t)

    # make inverse
    env = np.exp(t * L / T)
    s_t_flipped = s_t[::-1]
    s_t_inv = s_t_flipped / env

    return s_t, s_t_inv











