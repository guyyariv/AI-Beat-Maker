import scipy
import scipy.io.wavfile as wav
import scipy.signal as signal
from matplotlib import pyplot as plt
import numpy as np
import IPython.display as ipd
import librosa
import soundfile as sf
from scipy.ndimage import filters
import libfmp.b
import libfmp.c2
import libfmp.c6
import utils
import os, sys
from numba import jit
from beat_tracking.onset_detection import OnsetDetection

sys.path.append('..')


def beat_tracking(sec_duration, onset_envelope, tempo_estimation=None):
    if tempo_estimation == None:
        tempo_estimation = librosa.beat.tempo(onset_envelope=onset_envelope,
                                              aggregate=None)
    if tempo_estimation[np.insert(np.diff(tempo_estimation).astype(
            np.bool), 0, True)].size < sec_duration / 4:
        (_, idx, counts) = np.unique(tempo_estimation, return_index=True, return_counts=True)
        index = idx[np.argmax(counts)]
        mode = tempo_estimation[index]
        return librosa.beat.beat_track(onset_envelope=onset_envelope, bpm=mode,
                                       units='time')
    else:
        pulse = librosa.beat.plp(onset_envelope=onset_envelope)
        beats_plp = np.flatnonzero(librosa.util.localmax(pulse))
        beats = librosa.times_like(pulse)
        return tempo_estimation, beats[beats_plp]



if __name__ == '__main__':
    x, sr = librosa.load('reports/keys-of-moon-white-petals.wav')
    onset_detection = OnsetDetection()
    onset_envelope, fs = onset_detection.spectral_based_novelty(x, sr, H=512,
                                                                M=100,
                                                                norm=False)
    tempo, peaks_sec = beat_tracking(x.size / sr, onset_envelope)
    # tempo, peaks_sec = librosa.beat.beat_track(x, sr, units='time')
    x_peaks = librosa.clicks(peaks_sec, sr=sr, click_freq=1000, length=len(x))
    sf.write('test_guy2.wav', x + x_peaks, sr)
