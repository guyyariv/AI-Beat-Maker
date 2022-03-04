import numpy as np
import librosa
import utils
import os, sys
sys.path.append('..')


def beat_tracking(audio_data, sample_rate, onset_env=None, hop_length=512, start_bpm=120.0):
    if not onset_env:
        onset_env = librosa.onset.onset_strength(y=audio_data, sr=sample_rate, hop_length=hop_length)
    times = librosa.times_like(onset_env, sr=sample_rate)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, hop_length=hop_length, start_bpm=start_bpm)
    # utils.plot_reg_and_shuff_graphs(onset_env, beats)
    return tempo, times[beats]


def slice_by_beat_tracking(audio_data, sample_rate, onset_env=None, hop_length=512, start_bpm=120.0, music_frame=8):
    if not onset_env:
        onset_env = librosa.onset.onset_strength(y=audio_data, sr=sample_rate, hop_length=hop_length)
    times = librosa.times_like(onset_env, sr=sample_rate)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, hop_length=hop_length, start_bpm=start_bpm)
    peaks = beats[::music_frame]
    peak_vals = (times[peaks] * sample_rate).astype(int)
    slices = np.split(audio_data, peak_vals)[1:-1]
    return tempo, slices


def plp(audio_data, sample_rate, onset_env=None, hop_length=512, tempo_min=120, tempo_max=300, clicks=[]):
    if not onset_env:
        onset_env = librosa.onset.onset_strength(y=audio_data, sr=sample_rate, hop_length=hop_length)
    times = librosa.times_like(onset_env, sr=sample_rate)
    pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sample_rate, tempo_min=tempo_min, tempo_max=tempo_max)
    beats_plp = np.flatnonzero(librosa.util.localmax(pulse))
    m_plp = np.mean(onset_env[beats_plp])
    above_avg_plp = beats_plp[onset_env[beats_plp] >= m_plp]
    inner_peaks = np.delete(above_avg_plp, np.nonzero(np.in1d(above_avg_plp, clicks))[0])
    return times[inner_peaks]

