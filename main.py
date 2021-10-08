import scipy
import scipy.io.wavfile as wav
import scipy.signal as signal
from matplotlib import pyplot as plt
import numpy as np
import IPython.display as ipd
import librosa
import librosa.display
import soundfile as sf
from beat_tracking.onset_detection import OnsetDetection

if __name__ == '__main__':
    x, sr = librosa.load('resources/test.wav')
    onset_detection = OnsetDetection()
    pass
