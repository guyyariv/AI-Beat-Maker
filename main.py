import scipy
import scipy.io.wavfile as wav
import scipy.signal as signal
from matplotlib import pyplot as plt
import numpy as np
import IPython.display as ipd
import librosa
import librosa.display
import soundfile as sf


def specgram(filename):
    sample_rate, samples = wav.read(filename)
    f, t, Zxx = signal.stft(samples, fs=sample_rate)
    plt.pcolormesh(t, f, np.abs(Zxx))
    plt.specgram(samples, Fs=sample_rate)
    plt.show()


def beat_tracking(filename):
    x, sr = librosa.load(filename)
    ipd.Audio(x, rate=sr)
    tempo, beat_times = librosa.beat.beat_track(x, sr=sr, start_bpm=60,
                                                units='samples')
    plt.figure(figsize=(14, 5))
    x_ = x[beat_times[0]:beat_times[2]]
    librosa.display.waveplot(x_, alpha=0.6)
    plt.show()
    sf.write('reports/new.wav', x_, sr)


if __name__ == '__main__':
    beat_tracking('resources/file_example_WAV_10MG.wav')
