import scipy
import scipy.io.wavfile as wav
import scipy.signal as signal
from matplotlib import pyplot as plt
import numpy as np
import IPython.display as ipd
import librosa
import soundfile as sf
import random
import utils
import os, sys
from beat_tracking.onset_detection import OnsetDetection

sys.path.append('..')


def beat_tracking(audio_data, sample_rate, onset_env=None, hop_length=512, start_bpm=120.0):
    if not onset_env:
        onset_env = librosa.onset.onset_strength(y=audio_data, sr=sample_rate, hop_length=hop_length)
    times = librosa.times_like(onset_env, sr=sample_rate)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, hop_length=hop_length, start_bpm=start_bpm)
    return tempo, times[beats]


def slice_by_beat_tracking(audio_data, sample_rate, onset_env=None, hop_length=512, start_bpm=120.0, music_frame=4):
    if not onset_env:
        audio_data = audio_data[np.where(audio_data > 0.01)[0][0]:]
        onset_env = librosa.onset.onset_strength(y=audio_data, sr=sample_rate, hop_length=hop_length)
    times = librosa.times_like(onset_env, sr=sample_rate)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, hop_length=hop_length, start_bpm=start_bpm)
    peaks = beats[::music_frame]
    peak_vals = (times[peaks] * sample_rate).astype(int)
    slices = np.split(audio_data, peak_vals)
    return tempo, slices


def plp(audio_data, sample_rate, onset_env=None, hop_length=512, tempo_min=120, tempo_max=300):
    if not onset_env:
        onset_env = librosa.onset.onset_strength(y=audio_data, sr=sample_rate, hop_length=hop_length)
    times = librosa.times_like(onset_env, sr=sample_rate)
    pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sample_rate, tempo_min=tempo_min, tempo_max=tempo_max)
    beats_plp = np.flatnonzero(librosa.util.localmax(pulse))
    return times[beats_plp]

