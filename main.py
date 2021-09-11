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
                                                units='time')
    print(tempo)
    print(beat_times)
    plt.figure(figsize=(14, 5))
    x = np.fft.fftshift(x)
    parts = np.where(x > 0.9)
    x = x[parts[0][0]:parts[0][1]]
    librosa.display.waveplot(x, alpha=0.6)
    plt.show()
    # plt.vlines(beat_times, -1, 1, color='r')
    # plt.show()
    # plt.ylim(-1, 1)
    # plt.show()
    beat_times_diff = np.diff(beat_times)
    plt.figure(figsize=(14, 5))
    plt.hist(beat_times_diff, bins=50, range=(0, 4))
    plt.xlabel('Beat Length (seconds)')
    plt.ylabel('Count')
    plt.show()
    x = np.fft.ifftshift(x)
    sf.write('reports/new.wav', x, sr)


if __name__ == '__main__':
    beat_tracking('resources/test.wav')
