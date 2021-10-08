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


if __name__ == '__main__':
    x, sr = librosa.load('resources/test.wav')
    onset_detection = OnsetDetection()
    pass