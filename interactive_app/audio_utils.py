# Handle playback, noise gen, mic input

# Relative path definition
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))


import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve


import base64
import io
import sounddevice as sd




def load_audio(filepath):
    rate, data = wavfile.read(filepath)
    return rate, data

def apply_hrtf(audio, hrtf_left, hrtf_right):
    left = fftconvolve(audio, hrtf_left, mode='full')
    right = fftconvolve(audio, hrtf_right, mode='full')
    return np.stack([left, right], axis=1)




def decode_uploaded_audio(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    buffer = io.BytesIO(decoded)
    rate, data = wavfile.read(buffer)
    return data.astype(np.float32), rate

def play_audio_array(audio_stereo, rate):
    sd.play(audio_stereo, samplerate=rate)















