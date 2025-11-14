import os
import scipy.io
import numpy as np

import sys
# Go up 2 levels: from /hrtf_eval/notebooks to /my_project
project_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from hrtf_eval_py.utils.bp_filter import bp_filter


def load_hrirs_total(folderpath):
    try:
        mat_file_path = os.path.join(folderpath, 'data_hrir_total.mat')
        data_loaded = scipy.io.loadmat(mat_file_path, struct_as_record=False, squeeze_me=True)
        data = data_loaded['data_hrirs_total']

        IR_data = data.Data_IR
        fs = data.SamplingRate
        SourcePositions = data.SourcePositions
        ReceiverPosition = data.ReceiverPosition
        ListenerPosition = data.ListenerPosition

        return IR_data, fs, SourcePositions, ReceiverPosition, ListenerPosition

    except KeyError as e:
        print(f"Missing key in .mat file: {e}")
        return None
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return None


def load_hrirs_from_logs(folderpath):
    # Pre-initialize with empty arrays of appropriate types
    HRIR_L_BP_concat = np.empty((0, 0))
    HRIR_R_BP_concat = np.empty((0, 0))
    positions_concat = np.empty((0, 0))
    positions_concat_az = np.empty((0,))
    positions_az = np.empty((0,))
    positions_concat_rec = np.empty((0, 0))
    positions_rec_az = np.empty((0,))
    noise_floor_left_concat = np.empty((0, 0))
    noise_floor_right_concat = np.empty((0, 0))
    measurement_settings = {}
    noise_tail_left_concat = np.empty((0, 0))
    noise_tail_right_concat = np.empty((0, 0))
    fs = None

    # === RAW RECORDING LOGS FIRST to get fs ===
    try:
        raw_recording_logs_path = os.path.join(folderpath, 'raw_recording_logs')
        mat_files_rec = [f for f in os.listdir(raw_recording_logs_path) if f.endswith('.mat')]
        mat_files_rec.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

        positions_list_rec = []
        azimuth_list = []
        noise_left_list = []
        noise_right_list = []

        total_rec = len(mat_files_rec)
        for i, mat_file in enumerate(mat_files_rec):
            mat_data = scipy.io.loadmat(os.path.join(raw_recording_logs_path, mat_file), struct_as_record=False, squeeze_me=True)
            data_recordings_raw = mat_data['data_recordings_raw']

            positions_list_rec.append(data_recordings_raw.positions)
            azimuth_list.append(data_recordings_raw.positions[0, 0])
            noise_left_list.append(data_recordings_raw.noise_floor[:, 0])
            noise_right_list.append(data_recordings_raw.noise_floor[:, 1])

            if not measurement_settings:
                measurement_settings = {
                    'overlapRatio': data_recordings_raw.overlapRatio,
                    'playSeqLength': data_recordings_raw.playSeqLength,
                    'fs': data_recordings_raw.fs,
                    'f_lo': data_recordings_raw.f_lo,
                    'f_hi': data_recordings_raw.f_hi,
                    'play_ch': data_recordings_raw.play_ch,
                    'rec_ch': data_recordings_raw.rec_ch,
                    'measurement_method': data_recordings_raw.measurement_method
                }
                fs = data_recordings_raw.fs

            print(f"Recording logs reading: {((i + 1) / total_rec) * 100:.1f}%")

        if positions_list_rec:
            positions_concat_rec = np.vstack(positions_list_rec)
            positions_rec_az = np.array(azimuth_list)
            noise_floor_left_concat = np.column_stack(noise_left_list)
            noise_floor_right_concat = np.column_stack(noise_right_list)
    except Exception as e:
        print(f"Error loading raw recording logs: {e}")

    # === HRIR LOGS ===
    try:
        hrir_logs_path = os.path.join(folderpath, 'hrir_logs')
        mat_files_hrir = [f for f in os.listdir(hrir_logs_path) if f.endswith('.mat')]
        mat_files_hrir.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

        HRIR_L_BP_list = []
        HRIR_R_BP_list = []
        positions_list = []
        azimuth_list_hrir = []
        noise_tail_left_list = []
        noise_tail_right_list = []

        total_hrir = len(mat_files_hrir)
        for i, mat_file in enumerate(mat_files_hrir):
            mat_data = scipy.io.loadmat(os.path.join(hrir_logs_path, mat_file), struct_as_record=False, squeeze_me=True)
            data_hrirs_calculated = mat_data['data_hrirs_calculated']

            HRIR_L_BP_list.append(data_hrirs_calculated.HRIR_L_BP.T)
            HRIR_R_BP_list.append(data_hrirs_calculated.HRIR_R_BP.T)
            positions_list.append(data_hrirs_calculated.positions)
            azimuth_list_hrir.append(data_hrirs_calculated.positions[:, 0])

            # Noise Tail Extraction
            hrir_length = data_hrirs_calculated.measurement_struct.hrir_length
            playSeqLength = data_hrirs_calculated.measurement_struct.playSeqLength
            if fs is None:
                fs = 44100  # fallback to 1 if fs was not obtained from recordings
            constant_delay = round(playSeqLength * fs)
            crop_time_offset = 22050 + constant_delay # -256 added for debugging // initial version cropping from the beggining: problem is that the linear deconv pushes harmonics to the begining in IR, so let's crop noise from the end

            # idx2 = data_hrirs_calculated.sequence_chunk_indices[0, 0] + crop_time_offset # initial version 
            # idx1 = idx2 - hrir_length # initial version
            
            # idx1 = data_hrirs_calculated.sequence_chunk_indices[-1, 1] + crop_time_offset # initial version: stable
            idx1 = data_hrirs_calculated.sequence_chunk_indices[-1, 1] + hrir_length*1 # Debug version: Not yet approved stable
            idx2 = idx1 + hrir_length # initial version
            # print('Last chunk: ', data_hrirs_calculated.sequence_chunk_indices[-1, 1])
            # print('Noise index1: ', idx1)
            # print('Noise index2: ', idx2)

            imp_resp = data_hrirs_calculated.imp_resp
            imp_resp = bp_filter(imp_resp, 300, 18000, 'fft', 44100)
            noise_tail = imp_resp[idx1:idx2, :]  # shape: (hrir_length, 2)

            if noise_tail.shape[0] != hrir_length:
                raise ValueError(f"Noise tail length mismatch: got {noise_tail.shape[0]}, expected {hrir_length}")

            noise_tail_left_list.append(noise_tail[:, 0])
            noise_tail_right_list.append(noise_tail[:, 1])

            print(f"HRIR logs reading: {((i + 1) / total_hrir) * 100:.1f}%")

        if HRIR_L_BP_list:
            HRIR_L_BP_concat = np.vstack(HRIR_L_BP_list)
            HRIR_R_BP_concat = np.vstack(HRIR_R_BP_list)
            positions_concat = np.vstack(positions_list)
            positions_concat_az = np.concatenate(azimuth_list_hrir)
            positions_az = positions_concat_az.copy()
            noise_tail_left_concat = np.column_stack(noise_tail_left_list)
            noise_tail_right_concat = np.column_stack(noise_tail_right_list)
    except Exception as e:
        print(f"Error loading HRIR logs: {e}")

    return (
        HRIR_L_BP_concat,
        HRIR_R_BP_concat,
        positions_concat,
        positions_concat_az,
        positions_concat_rec,
        positions_rec_az,
        noise_floor_left_concat,
        noise_floor_right_concat,
        measurement_settings,
        noise_tail_left_concat,
        noise_tail_right_concat,
        positions_az
    )




# Example usage
if __name__ == "__main__":
    folderpath = "Y:/projects/Payman/matlab_workspace/AIP_HRTF_measurement_GUI_v4.0/data_measurement/data_testing_functionalityToAnalyze_datalog_20250528_170751_subject_arc_test"

    (
        HRIR_L_BP_concat,
        HRIR_R_BP_concat,
        positions_concat,
        positions_concat_az,
        positions_concat_rec,
        positions_rec_az,
        noise_floor_left_concat,
        noise_floor_right_concat,
        measurement_settings,
        noise_tail_left_concat,
        noise_tail_right_concat,
        positions_az
    ) = load_hrirs_from_logs(folderpath)

    print("HRIR_L_BP_concat shape:", None if HRIR_L_BP_concat is None else HRIR_L_BP_concat.shape)
    print("HRIR_R_BP_concat shape:", None if HRIR_R_BP_concat is None else HRIR_R_BP_concat.shape)
    print("positions_concat shape:", None if positions_concat is None else positions_concat.shape)
    print("positions_concat_az shape:", None if positions_concat_az is None else positions_concat_az.shape)
    print("positions_az shape:", None if positions_az is None else positions_az.shape)
    print("noise_tail_left_concat shape:", None if noise_tail_left_concat is None else noise_tail_left_concat.shape)
    print("noise_tail_right_concat shape:", None if noise_tail_right_concat is None else noise_tail_right_concat.shape)
    print("positions_concat_rec shape:", None if positions_concat_rec is None else positions_concat_rec.shape)
    print("positions_rec_az shape:", None if positions_rec_az is None else positions_rec_az.shape)
    print("noise_floor_left_concat shape:", None if noise_floor_left_concat is None else noise_floor_left_concat.shape)
    print("noise_floor_right_concat shape:", None if noise_floor_right_concat is None else noise_floor_right_concat.shape)
    print("Measurement settings:", measurement_settings)


    print("\nLoading data_hrir_total.mat...")
    result = load_hrirs_total(folderpath)

    if result is not None:
        IR_data, fs, SourcePositions, ReceiverPosition, ListenerPosition = result
        print("IR_data shape:", IR_data.shape if hasattr(IR_data, 'shape') else "Unknown")
        print("Sampling rate:", fs)
        print("SourcePositions shape:", SourcePositions.shape)
        print("ReceiverPosition:", ReceiverPosition)
        print("ListenerPosition:", ListenerPosition)
    else:
        print("Failed to load data_hrir_total")


