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


def beat_tracking(onset_envelope, tempo_estimation=None):
    if tempo_estimation == None:
        tempo_estimation = librosa.beat.tempo(onset_envelope=onset_envelope,
                                              aggregate=None)
    if np.mean(tempo_estimation) * 2 > np.max(tempo_estimation) and \
            np.mean(tempo_estimation) / 2 < np.min(tempo_estimation):
        return librosa.beat.beat_track(onset_envelope=onset_envelope,
                                               start_bpm=np.mean(
                                                   tempo_estimation),
                                               units='samples')
    else:
        pulse = librosa.beat.plp(onset_envelope=onset_envelope,
                                 tempo_max=np.max(tempo_estimation),
                                 tempo_min=np.min(tempo_estimation))
        beats_plp = np.flatnonzero(librosa.util.localmax(pulse))
        beats = librosa.samples_like(pulse)
        return np.mean(tempo_estimation), beats[beats_plp]


if __name__ == '__main__':
    x, sr = librosa.load('resources/beach.wav')
    onset_envelope = OnsetDetection().spectral_based_novelty(x, sr)
    beat_tracking(onset_envelope)