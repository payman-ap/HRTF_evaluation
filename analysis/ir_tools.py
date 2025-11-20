def time_align_ir(ir, sample_rate, expected_peak_search_window=None):
    """
    Find the main peak in the deconvolved impulse response and circularly shift
    so that the main peak is at index 0 (i.e. time 0). Works for mono or multi-channel.
    
    Parameters
    ----------
    ir : ndarray, shape (channels, N) or (N,)
        Deconvolved impulse responses (output of recorded * inverse).
    sample_rate : int
        Samples per second.
    expected_peak_search_window : tuple (t_min, t_max) in seconds, optional
        If provided, only search for the peak within [t_min, t_max] (helps avoid
        spurious early peaks). If None, search whole signal.
    
    Returns
    -------
    ir_aligned : ndarray same shape as input
        Circularly shifted so main peak is at sample index 0.
    peak_index : int
        Original index of the detected peak (before shifting).
    """
    ir = np.atleast_2d(ir)
    channels, N = ir.shape

    # compute envelope (sum-of-channels or max) to find global peak robustly
    env = np.max(np.abs(ir), axis=0)

    # define search range in samples
    if expected_peak_search_window is not None:
        tmin, tmax = expected_peak_search_window
        i_min = max(0, int(np.floor(tmin * sample_rate)))
        i_max = min(N - 1, int(np.ceil(tmax * sample_rate)))
    else:
        i_min, i_max = 0, N - 1

    # find index of max envelope within window
    local_env = env[i_min:i_max + 1]
    if local_env.size == 0:
        peak_rel = np.argmax(env)
        peak_index = peak_rel
    else:
        peak_rel = np.argmax(local_env)
        peak_index = i_min + int(peak_rel)

    # circular shift so peak_index becomes 0
    ir_aligned = np.roll(ir, -peak_index, axis=1)

    # If you prefer linear shift (pad with zeros instead of roll), uncomment:
    # shift = -peak_index
    # ir_aligned = np.zeros_like(ir)
    # if shift >= 0:
    #     ir_aligned[:, shift:] = ir[:, :N-shift]
    # else:
    #     ir_aligned[:, :N+shift] = ir[:, -shift:]

    # if input was 1D, return 1D
    if ir_aligned.shape[0] == 1:
        return ir_aligned[0], peak_index
    return ir_aligned, peak_index

#
#  Based on Farina's paper in 2007
#
def extract_harmonic_ir(ir_aligned, sample_rate, f1, f2, sweep_duration, max_harmonic=5,
                        window_ms=10, offset_after_peak_ms=0):
    """
    Given a time-aligned deconvolved IR (main linear IR at index 0), extract
    separate impulse responses for harmonic orders 1..max_harmonic.
    
    Parameters
    ----------
    ir_aligned : ndarray, shape (channels, N) or (N,)
        Time-aligned IR (peak at index 0).
    sample_rate : int
        Sampling rate in Hz.
    f1 : float
        Sweep start frequency (Hz).
    f2 : float
        Sweep stop frequency (Hz).
    sweep_duration : float
        Sweep duration in seconds (T).
    max_harmonic : int
        Number of harmonics to extract (2..max_harmonic).
    window_ms : float
        Window length (ms) used to capture each impulse response around the nominal
        harmonic position. Choose long enough to capture filter decay.
    offset_after_peak_ms : float
        Extra offset (ms) added to harmonic positions (sometimes used to
        ensure extraction window doesn't clip the impulse).
    
    Returns
    -------
    harmonic_ir_list : list of ndarray
        List of arrays: harmonic_ir_list[0] is the linear IR (m=1),
        harmonic_ir_list[1] is the 2nd harmonic IR, ..., up to max_harmonic.
        Each array shape: (channels, win_samples) or (win_samples,) for mono input.
    positions_samples : dict
        Mapping m -> start_sample index (in aligned IR) for that harmonic window.
    """
    ir = np.atleast_2d(ir_aligned)
    channels, N = ir.shape

    L = np.log(f2 / f1)
    T = float(sweep_duration)

    # window length in samples
    win_samples = int(np.ceil(window_ms * 1e-3 * sample_rate))
    half_win = win_samples // 2

    # compute nominal times for harmonics relative to 'end' of sweep:
    # t_m = (T / L) * ln(m)
    # After our alignment (main linear IR put at sample 0), the harmonic
    # components appear *before* the main IR. In the aligned IR, harmonic m
    # center index = - round(t_m * fs), so we add +offset to keep indices positive.
    # To make indices non-negative, we'll add an offset of win_samples or so.
    positions_samples = {}
    harmonic_ir_list = []

    # We will place linear IR (m=1) centered at 0 (we'll extract starting at 0)
    # But because we cannot have negative indices, we'll shift extraction start indices
    # by a safety offset.
    safety_offset = int(np.ceil(0.5 * sweep_duration * sample_rate))  # big enough padding
    # Create a padded IR so we can index negatively by shifting indices to +safety_offset
    padded = np.concatenate([np.zeros((channels, safety_offset)), ir, np.zeros((channels, safety_offset))], axis=1)

    for m in range(1, max_harmonic + 1):
        t_m = (T / L) * np.log(m)  # seconds
        # center sample in the aligned array (peak at index 0) is at negative t_m:
        center_idx_aligned = -int(round(t_m * sample_rate))  # negative or zero
        # convert to index in padded array
        center_idx_padded = safety_offset + center_idx_aligned + int(round(offset_after_peak_ms * 1e-3 * sample_rate))

        start = int(center_idx_padded - half_win)
        end = start + win_samples
        # clamp
        start = max(0, start)
        end = min(padded.shape[1], end)
        segment = padded[:, start:end]

        # if mono input originally, return 1D
        if segment.shape[0] == 1:
            segment = segment[0]
        harmonic_ir_list.append(segment)
        positions_samples[m] = (start - safety_offset)  # position relative to aligned IR (can be negative)

    return harmonic_ir_list, positions_samples


# ---------------------------
# Combined convenience helper
# ---------------------------
def align_and_extract(ir_deconvolved, sample_rate, f1, f2, sweep_duration,
                      max_harmonic=5, window_ms=10, expected_peak_search_window=None):
    """
    Align deconvolved IR and extract harmonic impulse responses.
    Returns:
      - ir_aligned (channels x N)
      - harmonic_ir_list (list: m=1..max_harmonic)
      - positions_samples (dictionary m->start index relative to aligned IR)
      - peak_index (original index of detected peak in deconvolved IR)
    """
    ir_deconvolved = np.atleast_2d(ir_deconvolved)
    ir_aligned, peak_index = time_align_ir(ir_deconvolved, sample_rate, expected_peak_search_window)
    harmonic_ir_list, positions = extract_harmonic_ir(ir_aligned, sample_rate, f1, f2, sweep_duration,
                                                      max_harmonic=max_harmonic, window_ms=window_ms)
    return ir_aligned, harmonic_ir_list, positions, peak_index









