import numpy as np

import os
import sys

# Go up 2 levels: from /hrtf_eval/analysis to /my_project
project_root = os.path.abspath(os.path.join(os.getcwd(), '..','..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from hrtf_eval_py.analysis.spectrum import real_fft_mag





# def hrir_to_hrtf_tmp(hrir_matrix, fs):

#     # hrir_matrix has a size of [2664 x 2 x 256]
#     for _, ir_ears in enumerate(hrir_matrix):
#         for _, ir_single_ear in enumerate(ir_ears):
#             tf_single_ear = real_fft_mag(ir_single_ear, fs)

#             # concatenate single ear tf to stereo ears
#             # tf_ears = 
#         # concatenate stereo ears to the whole hrtf data

#     # hrtf_matrix: expected to be 2664 x 2 x freqs length
#     return freqs, hrtf_matrix



def hrir_to_hrtf(hrir_matrix, fs, db_scale=False, n_fft=None):
    num_positions, num_ears, num_samples = hrir_matrix.shape
    
    # Get frequency axis from any FFT call
    freqs, _ = real_fft_mag(hrir_matrix[0, 0], fs, n_fft = n_fft)
    num_freqs = len(freqs)
    
    # Initialize HRTF matrix: [positions x ears x freqs]
    hrtf_matrix = np.zeros((num_positions, num_ears, num_freqs))
    
    for pos_idx, ir_ears in enumerate(hrir_matrix):
        for ear_idx, ir_single_ear in enumerate(ir_ears):
            _, mag = real_fft_mag(ir_single_ear, fs, db_scale=db_scale, n_fft=n_fft)  # just magnitude
            hrtf_matrix[pos_idx, ear_idx, :] = mag

    return freqs, hrtf_matrix