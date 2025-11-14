import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..'))) 
from hrtf_eval_py.analysis.ir_calculations import calculate_impulse_response
from hrtf_eval_py.utils.bp_filter import bp_filter
from hrtf_eval_py.utils.motu_functions_equivalent import motu_rms
from hrtf_eval_py.utils.window_funcs import gaussmod, window_gen
from hrtf_eval_py.utils.data_saver import quick_save_for_matlab
from hrtf_eval_py.utils.motu_functions_equivalent import motu_noise

import numpy as np
from scipy.signal.windows import hann, kaiser
from scipy.signal import lfilter, filtfilt
from scipy.fft import fft, ifft, fftshift



def db2mag(db):
    return 10 ** (db / 20)
def minimum_phase_from_magnitude(mag_spectrum):
    log_mag = np.log(np.maximum(mag_spectrum, 1e-8))
    cep = np.fft.ifft(log_mag).real
    cep[1:len(cep)//2] *= 2
    cep[len(cep)//2+1:] = 0
    min_phase_spec = np.fft.fft(np.exp(np.fft.fft(cep).real))
    return min_phase_spec
def smooth_target_band(freqs, f_low, f_high, transition=0.1):
    """Smoothly transition between 0 and 1 in passband."""
    mask = np.zeros_like(freqs)
    passband = (freqs >= f_low) & (freqs <= f_high)
    
    # Smooth transitions
    ramp_up = (freqs > f_low * (1 - transition)) & (freqs < f_low)
    ramp_down = (freqs > f_high) & (freqs < f_high * (1 + transition))
    
    mask[passband] = 1.0
    mask[ramp_up] = 0.5 * (1 + np.cos(np.pi * (freqs[ramp_up] - f_low) / (f_low * transition)))
    mask[ramp_down] = 0.5 * (1 + np.cos(np.pi * (freqs[ramp_down] - f_high) / (f_high * transition)))
    
    return mask



# def calculate_eq_filter(
#     ir,
#     fs=44100,
#     freq_range=(300, 18000),
#     win=None,
#     fft_len=32768,
#     smooth_kernel=None,
#     enable_phase_eq=False,
#     use_minimum_phase=False,
#     gain_norm_freq=1000
# ):
#     """
#     Calculate equalization filter from impulse response.
#     """
#     ir = np.asarray(ir).flatten()
#     if win is None:
#         win = hann(1024)
#     if smooth_kernel is None:
#         smooth_kernel = kaiser(256, 4)
#         smooth_kernel /= np.sum(smooth_kernel)

#     win_len = len(win)
#     ir_spec = fft(ir, fft_len)[:fft_len // 2 + 1]
#     freqs = np.linspace(0, fs / 2, fft_len // 2 + 1)

#     # Smoothed target magnitude response
#     target_spec = smooth_target_band(freqs, freq_range[0], freq_range[1])

#     eps = 1e-8
#     if enable_phase_eq:
#         # Complex inversion (preserve phase)
#         eq_spec = target_spec / (ir_spec + eps)
#     else:
#         # Magnitude inversion only
#         mag_ir_spec = np.abs(ir_spec)
#         inv_mag = target_spec / (mag_ir_spec + eps)

#         # Optional smoothing
#         inv_mag = np.sqrt(np.convolve(inv_mag**2, smooth_kernel, mode='same'))

#         if use_minimum_phase:
#             eq_spec = minimum_phase_from_magnitude(inv_mag)
#         else:
#             eq_spec = inv_mag

#     # Symmetric spectrum for full IFFT
#     eq_spec_full = np.concatenate([eq_spec, np.conj(eq_spec[-2:0:-1])])
#     eq_filter_full = np.real(ifft(eq_spec_full))

#     # Normalize gain at reference frequency
#     if gain_norm_freq:
#         ref_bin = np.argmin(np.abs(freqs - gain_norm_freq))
#         gain = np.abs(eq_spec[ref_bin])
#         eq_filter_full /= (gain + eps)

#     # Center and window
#     eq_filter_full = fftshift(eq_filter_full)
#     center = np.argmax(np.abs(eq_filter_full))
#     start = int(center - win_len // 2)
#     stop = start + win_len
#     eq_filter = eq_filter_full[start:stop] * win

#     eq_lag_sample = win_len // 2

#     return eq_filter, eq_lag_sample








def calculate_eq_filter(
    ir,
    fs=44100,
    freq_range=(300, 18000),
    win=None,
    fft_len=1024, # 32768
    smooth_kernel=None,
    enable_phase_eq=False
):
    """
    Calculate equalization filter from impulse response.

    Parameters:
        ir (1D np.ndarray): Impulse response.
        fs (float): Sampling frequency (Hz).
        freq_range (tuple): Equalization frequency range (Hz).
        win (1D np.ndarray): Window function for truncating the EQ filter.
        fft_len (int): FFT length for frequency resolution.
        smooth_kernel (1D np.ndarray): Kernel for smoothing.
        enable_phase_eq (bool): Enable phase correction mode.

    Returns:
        eq_filter (1D np.ndarray): Time-domain equalization filter.
        eq_lag_sample (int): Center sample index of the EQ filter.
    """
    ir = np.asarray(ir).flatten()
    if win is None:
        win = hann(512)
    if smooth_kernel is None:
        smooth_kernel = kaiser(128, 2)
        smooth_kernel /= np.sum(smooth_kernel)

    win_len = len(win)
    sig_len = len(ir)

    ir_spec = fft(ir, fft_len)[:fft_len // 2 + 1]
    freqs = np.linspace(0, fs / 2, fft_len // 2 + 1)

    target_spec = np.ones_like(ir_spec)
    target_spec[freqs <= freq_range[0]] = 0
    target_spec[freqs > freq_range[1]] = 0

    if enable_phase_eq:
        eps = 1e-8
        eq_spec = target_spec / (ir_spec + eps)
        eq_spec_full = np.concatenate([eq_spec, np.conj(eq_spec[-2:0:-1])])
        eq_filter_full = np.real(ifft(eq_spec_full))

        eq_filter_full = np.roll(eq_filter_full, -fft_len // 2)
        ir_max_idx = np.argmax(np.abs(eq_filter_full))

        eq_filter = np.zeros_like(eq_filter_full)
        start = int(ir_max_idx - win_len // 2)
        stop = int(ir_max_idx + win_len // 2)
        eq_filter[start:stop] = eq_filter_full[start:stop] * win

        eq_lag_sample = ir_max_idx

    else:
        eps = 1e-8
        eq_spec = target_spec / np.abs(ir_spec + eps)
        eq_spec_smooth = np.sqrt(np.convolve(eq_spec**2, smooth_kernel, mode='same'))

        eq_spec_full = np.concatenate([eq_spec_smooth, np.conj(eq_spec_smooth[-2:0:-1])])
        eq_filter_full = np.real(ifft(eq_spec_full))

        eq_filter_full = fftshift(eq_filter_full)
        ir_max_idx = np.argmax(np.abs(eq_filter_full))

        start = int(ir_max_idx - win_len // 2)
        stop = int(ir_max_idx + win_len // 2)
        eq_filter = eq_filter_full[start:stop] * win

        eq_lag_sample = win_len // 2

    return eq_filter, eq_lag_sample



def eq_filter_calc_from_playrec(recorded_signals, excitation_signal, fs, measure_level, eq_filter_length=1024, consider_mic_directionality=False, mic_directionality_filters=None, loudspeaker_placement=None):


    # Extra code to remove silence - to be removed in stable version
    sweep_duration_samples = int(5 * fs)  # 5 seconds sweep
    sweep = excitation_signal[:sweep_duration_samples]  # Extract only sweep part
    f1, f2 = 150, 20000

    impulse_responses = []

    # Calculate the IRs

    for ch in range(recorded_signals.shape[1]):
        recorded_ch = recorded_signals[:, ch]
        ir = calculate_impulse_response(sweep, recorded_ch, fs, f1, f2)
        impulse_responses.append(ir)

    impulse_responses = np.stack(impulse_responses, axis=0)

    # Compensate for microphone
    if consider_mic_directionality:
        pass

    # Estimate the delay from peak

    # Calculate the EQ filters
    # # Step 1: Estimate delay per channel (peak index)
    est_delay_samples = np.argmax(np.abs(impulse_responses), axis=1)
    min_channel_lag = np.min(est_delay_samples)

    # Containers
    eq_filters_time_compensated = []
    equalization_filters = []
    eq_filters_level_corrected = []
    verification_impulse_responses = []


    # Constants from MATLAB toolbox:
    win_len = 512
    filter_len = 1024 # eq_filter_length
    fft_len = 32718
    win_offset = 70
    win_ramp = 40
    sig = np.ones((512, 1))
    risetime_ms = win_ramp/44100*1000
    # _, win = gaussmod(sig, risetime_ms=risetime_ms, fs=44100)

    win = window_gen('hann', 50, 412, 50, symmetry_type=True)

    freq_range = [f1, f2]



    channel_num = impulse_responses.shape[0]





    # Loop over each channel
    for kk in range(channel_num):
        ir = impulse_responses[kk, :]
        ir_peak_idx = est_delay_samples[kk]

        if ir_peak_idx < win_len // 2:
            raise ValueError(f"IR start sample less than half window length: {win_len // 2}")

        # Window the IR aligned to the earliest detected peak
        ir_win = np.zeros(filter_len)
        start_idx_ir = ir_peak_idx + win_offset - win_len // 2
        end_idx_ir = ir_peak_idx + win_offset + win_len // 2

        start_idx_win = ir_peak_idx - min_channel_lag + filter_len // 2 - win_len // 2
        end_idx_win = ir_peak_idx - min_channel_lag + filter_len // 2 + win_len // 2

        ir_win[start_idx_win:end_idx_win] = ir[start_idx_ir:end_idx_ir] * win

        # Compute the equalization filter
        eq_filter, _ = calculate_eq_filter(
            ir_win,
            fs=fs,
            freq_range=freq_range,
            fft_len=fft_len,
            win=win,
            smooth_kernel=np.array([1.0]),  # Equivalent to MATLAB's '1'
            enable_phase_eq=True
        )

        # Extract time-compensated EQ filter from middle
        start_idx_eq = fft_len // 2 - filter_len
        end_idx_eq = fft_len // 2

        eq_filter_shift = eq_filter[start_idx_eq:end_idx_eq]

        equalization_filters.append(eq_filter)
        eq_filters_time_compensated.append(eq_filter_shift)
        

    # Convert to arrays: shape [channels, filter_len]
    equalization_filters = np.stack(equalization_filters, axis=0)
    eq_filters_time_compensated = np.stack(eq_filters_time_compensated, axis=0)
    

    # equalization_filters_final = eq_filters_time_compensated



    # Compensate for delay

    # Compensate for level
    level_calibration_signal = motu_noise(measure_level, f1, f2, 5, t_rise=0, Fs=fs)

    for kk in range(channel_num):

        level_calibration_signal_tmp = lfilter(eq_filters_time_compensated[kk,:],[1.0], level_calibration_signal)

        level, _ = motu_rms(level_calibration_signal_tmp, wide=1)

        eq_filters_level_corrected_tmp = eq_filters_time_compensated[kk,:] * db2mag(measure_level-level)

        eq_filters_level_corrected.append(eq_filters_level_corrected_tmp)

    eq_filters_level_corrected = np.stack(eq_filters_level_corrected, axis=0)

    equalization_filters_final = eq_filters_level_corrected





    # Verify the EQ filters for level
    verification_test_signal = sweep
    level, _ = motu_rms(verification_test_signal, wide=1)
    verification_test_signal *= db2mag(measure_level-level)

    for kk in range(channel_num):

        verification_signal_mic_compensated = lfilter(mic_directionality_filters[kk,:], [1.0], verification_test_signal)

        verification_signal_equalized = lfilter(equalization_filters_final[kk,:], [1.0], verification_signal_mic_compensated)

        verification_signal_reconstructed = lfilter(impulse_responses[kk,:], [1.0], verification_signal_equalized)

        # verification_signal_reconstructed = bp_filter(verification_signal_reconstructed, f1, f2, method='fft')

        # Calculate Verification IRs

        verification_impulse_response_tmp = calculate_impulse_response(sweep, verification_signal_reconstructed, fs, f1, f2)


        verification_impulse_responses.append(verification_impulse_response_tmp)

        print(f"Channel {kk}/{channel_num} completed")

    verification_impulse_responses = np.stack(verification_signal_reconstructed, axis=0)








    # Convolve with IRs to simulate

    # Apply mic directivity


    # calculate level with motu_rms


    # compensate for level deviation of EQ filters db2mag

    #verify the IRs, conv check level




        








    
    # return EQ filter matrix



    return impulse_responses, equalization_filters_final, eq_filters_level_corrected, verification_impulse_responses







if __name__=="__main__":

    # load recording data
    file_path = r'Y:\projects\Payman\matlab_workspace\AIP_HRTF_measurement_GUI_v4.0\data_calibration_files\20250618_eqFilters_arc_afterRepair\eq_filters_20250618_103934.mat'
    mat_data = loadmat(file_path, struct_as_record=False, squeeze_me=True)
    # Access the top-level struct
    data = mat_data['data']
    # Access nested fields
    recSig = data.ampEq.recSig
    excitSig = data.ampEq.excitSig
    # print("recSig:", recSig.shape)
    # print("excitSig:", excitSig.shape)

    # load mic directionality matrix
    file_path_mic_dir = r'Y:\projects\Payman\matlab_workspace\AIP_HRTF_measurement_GUI_v4.0\data_calibration_files\mm210_dir_filters.mat'
    mat_mic_dir = loadmat(file_path_mic_dir)
    mic_dir_filters = mat_mic_dir['mic_dir_filters']
    # print("mic_dir_filters:", mic_dir_filters.shape)

    # load loudspeaker placement matrix
    file_path_ls_dir = r'Y:\projects\Payman\matlab_workspace\AIP_HRTF_measurement_GUI_v4.0\data_hardware\SpeakerElevations_data.mat'
    mat_ls_dir = loadmat(file_path_ls_dir)
    speaker_placement = mat_ls_dir['SpeakerElevations_data']
    # print("loudspeaker_dir:", speaker_placement.shape)


    # Calculating IRs
    impulse_responses, equalization_filters_final, eq_filters_level_corrected, verification_impulse_responses = eq_filter_calc_from_playrec(recSig, 
                                                                                                                                 excitSig, 44100, 70, 
                                                                                                                                 eq_filter_length=1024, 
                                                                                                                                 consider_mic_directionality=True, 
                                                                                                                                 mic_directionality_filters=mic_dir_filters.T, 
                                                                                                                                 loudspeaker_placement=None)
    print('impulse_responses:', impulse_responses.shape)
    # for i in range(32):
    #     plt.plot(impulse_responses[i,:])
    # plt.show()

    print('equalization_filters:', equalization_filters_final.shape)
    # for i in range(32):
    #     plt.plot(equalization_filters[i,:])
    # plt.show()

    quick_save_for_matlab(r'Y:\projects\Payman\matlab_workspace\AIP_HRTF_measurement_GUI_v4.0\data_calibration_files\20250618_eqFilters_arc_afterRepair\eq_from_py.mat',
                          impulse_responses=impulse_responses,
                          equalization_filters=equalization_filters_final, 
                          verification_impulse_responses=verification_impulse_responses)

    