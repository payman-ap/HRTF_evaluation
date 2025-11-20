import numpy as np
from scipy.fft import fft, ifft

def calculate_impulse_response(sweep, recorded, fs, f1, f2):
    """
    Calculate impulse response using deconvolution of a logarithmic sweep.
    
    Parameters:
        sweep (1D np.ndarray): Excitation sweep signal (only the sweep, no silence).
        recorded (1D np.ndarray): Recorded response (includes sweep + silence).
        fs (int or float): Sampling rate in Hz.
        f1 (float): Start frequency of the sweep.
        f2 (float): End frequency of the sweep.
        
    Returns:
        impulse_response (1D np.ndarray): Calculated impulse response.
    """
    # Duration of sweep in seconds
    sweep_duration = len(sweep) / fs
    
    # Time vector for the sweep
    t_sweep = np.linspace(0, sweep_duration, len(sweep), endpoint=False)
    
    # Exponential envelope correction
    env = np.exp(np.log(f2 / f1) * t_sweep / sweep_duration)
    
    # Generate inverse filter: time-reversed sweep divided by envelope
    inverse = sweep[::-1] / env
    
    # Zero-pad inverse to match recorded signal length
    pad_length = len(recorded) - len(inverse)
    if pad_length < 0:
        raise ValueError("Recorded signal is shorter than sweep; check input lengths.")
    inverse_padded = np.concatenate((inverse, np.zeros(pad_length)))
    
    # Length for FFT convolution
    N = len(recorded) + len(inverse_padded) - 1
    
    # Frequency domain deconvolution
    IR = ifft(fft(recorded, N) * fft(inverse_padded, N))
    IR = np.real(IR)
    
    return IR


def deconv(recorded, inverse):
    """
    Compute the impulse response by FFT deconvolution:
        IR = IFFT( FFT(recorded) * FFT(inverse) )

    Parameters
    ----------
    recorded : ndarray, shape (channels, samples)
        Recorded sweep response (your microphone recording).
    inverse : ndarray, shape (samples,)
        Inverse filter (already padded to match sweep length).

    Returns
    -------
    ir : ndarray, shape (channels, samples + inverse_samples - 1)
        The impulse responses for each channel.
    """

    # Ensure 2D shape: (channels, samples)
    recorded = np.atleast_2d(recorded)

    num_channels, len_recorded = recorded.shape
    len_inverse = len(inverse)

    # Convolution length
    N = len_recorded + len_inverse - 1

    # Precompute FFT of inverse filter
    fft_inverse = np.fft.fft(inverse, N)

    # Output IR array
    impulse_responses = np.zeros((num_channels, N))

    # Process each channel
    for ch in range(num_channels):
        fft_rec = np.fft.fft(recorded[ch], N)
        ir = np.fft.ifft(fft_rec * fft_inverse)
        impulse_responses[ch] = np.real(ir)

    return impulse_responses




