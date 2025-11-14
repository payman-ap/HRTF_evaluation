import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
# going up one folder level to see the package

# The entire evaluation pipeline orchestration
# from hrtf_eval_py.core import HRTFMeasurement if main.py is outside hrtf_eval_py folder
# Here using relative imports
from hrtf_eval_py.utils.window_funcs import window_gen


import matplotlib.pyplot as plt


def main():
    # using r as raw_string: not to interpret backslashes / as  escape characters
    path_to_data = r'Y:\projects\Payman\matlab_workspace\AIP_HRTF_measurement_GUI_v4.0\data_measurement\data_testing_functionalityToAnalyze_datalog_20250528_170751_subject_arc_test'
    # ir_set, fs, SourcePositions, ReceiverPosition, ListenerPosition = load_hrirs_total(path_to_data)


    WIN = window_gen('hann', 50, 412, 50, symmetry_type=False)
    plt.plot(WIN)
    plt.show()
    
    
    
    # Spatial plotting




    # making the HRTF data object to use the data easily
    # HRTF_data_obj = HRTFMeasurement(ir_set, fs, SourcePositions, ReceiverPosition, ListenerPosition)
    
    # freq, mag = compute_fft(HRTF_data_obj.get_directional_hrtf(0,0))
    
    # plot_spectrum(freq, mag, title=f"Spectrum @ Az {HRTF_data_obj.azimuth}")


if __name__ == "__main__":
    main()






