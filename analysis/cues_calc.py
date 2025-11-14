import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate

def itd_calculation(
    Data_IR,
    sampling_frequency,
    method='onset_detection',
    threshold_level=-10,
    plot_on_fail=True,
    freq_range=(1000, 15000)  # Only used for 'group_delay'
):
    """
    Calculate Interaural Time Differences (ITDs) and lags from HRIRs using various methods.

    Parameters:
        Data_IR : ndarray
            HRIR data with shape [Positions, 2, signal_length]
        sampling_frequency : float
            Sampling rate in Hz
        method : str
            'onset_detection', 'group_delay', or 'cross_correlation'
        threshold_level : float
            Threshold in dB for onset detection (default: -10)
        plot_on_fail : bool
            If True, plot signals where onset detection fails
        freq_range : tuple (min_freq, max_freq)
            Frequency range (Hz) for group delay analysis

    Returns:
        ITDs : ndarray
            Interaural Time Differences [Positions x 1] in seconds (right - left)
        LAGs : ndarray
            Lag times of [Left, Right] channels [Positions x 2] in seconds
    """

    num_positions = Data_IR.shape[0]
    ITDs = np.zeros((num_positions, 1))
    LAGs = np.zeros((num_positions, 2))

    for i in range(num_positions):
        left = Data_IR[i, 0, :]
        right = Data_IR[i, 1, :]

        if method == 'onset_detection':
            for ch, signal in enumerate([left, right]):
                max_amp = np.max(np.abs(signal))
                threshold_amp = max_amp * 10**(threshold_level / 20)
                onset_index = 0

                for j in range(len(signal) - 1):
                    if signal[j] <= threshold_amp and signal[j + 1] > threshold_amp:
                        onset_index = j + 1 if abs(signal[j]) > abs(signal[j + 1]) else j
                        break
                else:
                    if plot_on_fail:
                        print(f"[Warning] No onset found for position {i}, channel {ch}. Using index 0.")
                        plt.plot(signal)
                        plt.title(f"No onset at Position {i}, Channel {ch}")
                        plt.show()
                    onset_index = 0

                LAGs[i, ch] = onset_index / sampling_frequency

        elif method == 'group_delay':
            min_f, max_f = freq_range
            for ch, signal in enumerate([left, right]):
                n = len(signal)
                freqs = np.fft.fftfreq(n, d=1 / sampling_frequency)
                spectrum = np.fft.fft(signal)
                unwrap_phase = np.unwrap(np.angle(spectrum))

                # Avoid division by zero for zero frequency spacing
                delta_f = np.diff(freqs)
                delta_phi = np.diff(unwrap_phase)
                group_delay_values = -delta_phi / (2 * np.pi * delta_f)

                # Use only positive frequencies within the desired range
                freqs_central = freqs[1:]
                valid_mask = (freqs_central > min_f) & (freqs_central < max_f)
                valid_gd = group_delay_values[valid_mask]

                delay_sec = np.median(valid_gd) if valid_gd.size > 0 else 0.0
                LAGs[i, ch] = delay_sec

        # elif method == 'cross_correlation':
        #     xcorr = correlate(right, left, mode='full')
        #     lags = np.arange(-len(left) + 1, len(right))
        #     peak_lag = lags[np.argmax(xcorr)]
        #     ITDs[i] = peak_lag / sampling_frequency
        #     LAGs[i, :] = [0, ITDs[i]] if peak_lag >= 0 else [-ITDs[i], 0]
        #     continue  # Skip default ITD calc

        else:
            raise ValueError(f"Unknown method: {method}")

        ITDs[i] = LAGs[i, 1] - LAGs[i, 0]

    return ITDs, LAGs




from scipy.signal import butter, sosfiltfilt

def bandpass_filter(signal, fs, f_low, f_high, order=4):
    sos = butter(order, [f_low, f_high], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, signal)

def ild_calculation(
    Data_IR,
    sampling_frequency,
    method='rms_energy',
    bands='full',  # 'full', 'octave', or 'third_octave'
    band_limits=None
):
    """
    Calculate Interaural Level Differences (ILDs) from HRIR data, with optional frequency-band analysis.

    Parameters:
        Data_IR : ndarray
            HRIR data with shape [Positions, 2, signal_length]
        sampling_frequency : float
            Sampling rate in Hz
        method : str
            Method for ILD computation (currently only 'rms_energy' is supported)
        bands : str
            'full' = fullband, 'octave', or 'third_octave'
        band_limits : list of tuples
            List of (f_low, f_high) frequency ranges for filtering (optional; overrides default bands)

    Returns:
        ILDs : ndarray
            ILDs in dB. Shape:
              - [Positions x 1] for fullband
              - [Positions x N_bands] for band-specific
        band_edges : list of (f_low, f_high) tuples for each band (None if fullband)
    """

    num_positions = Data_IR.shape[0]
    signal_length = Data_IR.shape[2]

    # Default band definitions
    if band_limits is None and bands != 'full':
        center_freqs = []
        if bands == 'octave':
            center_freqs = [125, 250, 500, 1000, 2000, 4000, 8000]
        elif bands == 'third_octave':
            base = 1000 * 2**(-3/2)
            center_freqs = [base * 2**(i/3) for i in range(13)]  # 13 bands centered ~125 to 8kHz
        band_limits = [(f / 2**(1/2), f * 2**(1/2)) for f in center_freqs]

    if bands == 'full':
        ILDs = np.zeros((num_positions, 1))
        for i in range(num_positions):
            left = Data_IR[i, 0, :]
            right = Data_IR[i, 1, :]
            rms_left = np.sqrt(np.mean(left**2))
            rms_right = np.sqrt(np.mean(right**2))
            ILDs[i, 0] = 20 * np.log10(rms_right / (rms_left + 1e-12))
        band_edges = None

    else:
        n_bands = len(band_limits)
        ILDs = np.zeros((num_positions, n_bands))
        for i in range(num_positions):
            for b, (f_low, f_high) in enumerate(band_limits):
                left_filt = bandpass_filter(Data_IR[i, 0, :], sampling_frequency, f_low, f_high)
                right_filt = bandpass_filter(Data_IR[i, 1, :], sampling_frequency, f_low, f_high)
                rms_left = np.sqrt(np.mean(left_filt**2))
                rms_right = np.sqrt(np.mean(right_filt**2))
                ILDs[i, b] = 20 * np.log10(rms_right / (rms_left + 1e-12))
        band_edges = band_limits

    return ILDs, band_edges



from scipy.signal import savgol_filter, find_peaks

def detect_main_notches(hrtf_dB, freqs, n_notches=3, min_depth_dB=6, freq_range=(4000, 12000)):
    freqs = np.asarray(freqs)
    hrtf_dB = np.asarray(hrtf_dB)

    # Restrict to frequency range
    valid_idx = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    freqs_in = freqs[valid_idx]
    hrtf_in = hrtf_dB[valid_idx]

    # Smooth to suppress microstructure (optional step)
    hrtf_smooth = savgol_filter(hrtf_in, window_length=11, polyorder=3)

    # Detect valleys (negative peaks)
    peaks, props = find_peaks(-hrtf_smooth, prominence=min_depth_dB)

    # Collect notches: (depth, freq)
    notch_list = []
    for i, idx in enumerate(peaks):
        freq_val = freqs_in[idx]
        depth_val = props["prominences"][i]
        notch_list.append((depth_val, freq_val))

    # Sort by depth (descending) and keep top n_notches
    notch_list.sort(reverse=True)
    notch_list = notch_list[:n_notches]

    # Pad with zeros if fewer than n_notches
    while len(notch_list) < n_notches:
        notch_list.append((0.0, 0.0))

    # Separate into consistent arrays
    notches_mag = np.array([n[0] for n in notch_list])  # depth in dB
    notches_freq = np.array([n[1] for n in notch_list])  # frequency in Hz

    return notches_freq, notches_mag





import numpy as np
from scipy.signal import savgol_filter, find_peaks, peak_widths

def erb_hz(f_hz):
    # Glasberg & Moore (1990) ERB
    return 24.7 * (4.37 * (f_hz / 1000.0) + 1.0)

def freq_prior(f_hz, f0=8000.0, sigma=2000.0, band=(4000.0, 12000.0)):
    # Smooth prior favoring the mid-high pinna-cue region
    if (f_hz < band[0]) or (f_hz > band[1]):
        return 0.0
    return np.exp(-0.5 * ((f_hz - f0) / sigma) ** 2)

def octave_distance(f1, f2):
    # absolute distance in octaves
    return abs(np.log2(f1 / f2))

def detect_and_rank_notches_advanced(
    hrtf_dB, freqs,
    n_return=5,
    freq_range=(4000, 12000),
    min_depth_dB=6.0,
    sg_window=11, sg_poly=3,
    min_sep_oct=1/5,        # ~0.2 oct (~15%) ≈ 1/5 octave
    width_rel_height=0.5,   # half-prominence width
):
    freqs = np.asarray(freqs)
    H = np.asarray(hrtf_dB)

    # limit band
    band = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    f_in = freqs[band]
    H_in = H[band]
    if len(f_in) < sg_window:
        # fallback if very short
        sg_window = max(5, (len(f_in)//2)*2+1)  # nearest odd <= len

    # smooth (preserve notch positions)
    Hs = savgol_filter(H_in, window_length=sg_window, polyorder=sg_poly)

    # find valleys as peaks in -Hs, keep by prominence (depth in dB)
    peak_idx, props = find_peaks(-Hs, prominence=min_depth_dB)
    if len(peak_idx) == 0:
        return np.array([]), np.array([]), np.array([])

    # measure widths at given relative height on -Hs, returns width in samples
    widths_samp, w_l, w_r, _ = peak_widths(-Hs, peak_idx, rel_height=width_rel_height)

    # convert all metrics
    freqs_notch = f_in[peak_idx]
    depths_dB = props["prominences"]        # already in dB because we used dB spectrum
    # convert sample widths to Hz with local bin spacing
    # assume nearly uniform spacing within the limited band
    if len(f_in) > 1:
        bin_hz = np.mean(np.diff(f_in))
    else:
        bin_hz = 1.0
    widths_hz = widths_samp * bin_hz

    # compute psychoacoustic terms
    erb_vals = erb_hz(freqs_notch)
    width_fac = np.minimum(1.0, widths_hz / np.maximum(erb_vals, 1e-9))  # 0..1
    prior = np.array([freq_prior(f) for f in freqs_notch])

    # redundancy penalty: for each notch, if a deeper notch is within min_sep_oct, down-weight
    R = np.ones_like(depths_dB)
    # sort by depth descending to compare against deeper neighbors
    order = np.argsort(-depths_dB)
    for i_pos, i in enumerate(order):
        for j in order[:i_pos]:  # only deeper ones
            if octave_distance(freqs_notch[i], freqs_notch[j]) < min_sep_oct:
                R[i] = 0.5 * R[i]  # penalize; you can make this smoother if desired

    # final score
    wD = 1.0
    scores = wD * depths_dB * width_fac * prior * R

    # sort and return
    rank = np.argsort(-scores)
    idx_sel = rank[:n_return]
    return freqs_notch[idx_sel], depths_dB[idx_sel], scores[idx_sel]




