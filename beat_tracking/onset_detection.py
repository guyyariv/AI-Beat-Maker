import scipy.io.wavfile as wav
import scipy.signal as signal
from matplotlib import pyplot as plt
import numpy as np
import IPython.display as ipd
import librosa
import librosa.display
import soundfile as sf
from scipy.ndimage import filters
import libfmp.b
import libfmp.c2
import libfmp.c6
import utils
import os, sys
from scipy import signal
from numba import jit
sys.path.append('..')


class OnsetDetection:
    def __calculate_local_energy_function(self, x, window):
        """
        Calculate local energy function
        :param x (np.ndarray): Signal
        :param window
        :return: local energy function
        """
        energy_local = np.convolve(x ** 2, window ** 2, 'same')
        return energy_local

    def __calculate_logarithmic_compression(self, signal, gamma):
        """

        :param energy: local energy function
        :param gamma: Parameter for logarithmic compression
        :return: logarithmic compression
        """
        return np.log(1 + gamma * signal)

    def __calculate_discrete_derivative(self, energy):
        """
        :param energy: local energy function
        :return: discrete derivative
        """
        return np.diff(signal)

    def __compute_local_average(self, x, M):
        """
        Compute local average of signal
        :param x:
        :param M:
        :return:
        """
        L = len(x)
        local_average = np.zeros(L)
        for m in range(L):
            a = max(m - M, 0)
            b = min(m + M + 1, L)
            local_average[m] = (1 / (2 * M + 1)) * np.sum(x[a:b])
        return local_average

    def __principal_argument(self, value):
        """
        Principal argument function
        :param value (or vector of values)
        :return: Principle value of v
        """
        w = np.mod(value + 0.5, 1) - 0.5
        return w

    def __energy_based_novelty(self, x, Fs=1, N=2048, H=128, gamma=10.0,
                               norm=True):
        """
        Compute energy-based novelty function
        :param x (np.ndarray): Signal
        :param Fs: Sampling rate
        :param N: Window size
        :param H: Hop size
        :param gamma: Parameter for logarithmic compression
        :param norm (bool): Apply max norm
        :return: novelty_energy (np.ndarray): Energy-based novelty function
        Fs_feature (scalar): Feature rate
        """
        window = signal.windows.hann(N)
        Fs_feature = Fs / H

        energy_local = self.__calculate_local_energy_function(x, window)[::H]
        if gamma:
            energy_local = self.__calculate_logarithmic_compression(energy_local, gamma)
        discrete_derivative = self.__calculate_discrete_derivative(energy_local)

        # half wave rectification
        novelty_energy = np.copy(discrete_derivative)
        novelty_energy[discrete_derivative < 0] = 0

        if norm:
            max_value = max(novelty_energy)
            if max_value > 0:
                novelty_energy = novelty_energy / max_value
        return novelty_energy, Fs_feature

    def spectral_based_novelty(self, x, Fs=1, N=1024, H=256, gamma=100.0,
                               M=10, norm=True, librosa_window='hanning'):
        """
        Compute spectral-based novelty function
        :param x (np.ndarray): Signal
        :param Fs: Sampling rate
        :param N: Window size
        :param H: Hop size
        :param gamma: Parameter for logarithmic compression
        :param norm (bool): Apply max norm
        :param M: Size (frames) of local average
        :param librosa_window: window for stft
        :return: novelty_spectrum (np.ndarray): Energy-based novelty function
        Fs_feature (scalar): Feature rate
        """
        Fs_feature = Fs / H
        stft = librosa.stft(x, n_fft=N, hop_length=H, win_length=N,
                         window=librosa_window)
        X = np.abs(stft)
        if gamma:
            X = self.__calculate_logarithmic_compression(X, gamma)
        discrete_derivative = self.__calculate_discrete_derivative(X)

        # half wave rectification
        novelty_spectrum = np.copy(discrete_derivative)
        novelty_spectrum[discrete_derivative < 0] = 0
        novelty_spectrum = np.sum(novelty_spectrum, axis=0)
        novelty_spectrum = np.concatenate((novelty_spectrum, np.array([0.0])))
        if M > 0:
            local_average = self.__compute_local_average(novelty_spectrum, M)
            novelty_spectrum = novelty_spectrum - local_average
            novelty_spectrum[novelty_spectrum < 0] = 0.0
        if norm:
            max_value = max(novelty_spectrum)
            if max_value > 0:
                novelty_spectrum = novelty_spectrum / max_value
        return novelty_spectrum, Fs_feature

    def phase_based_novelty(self, x, Fs=1, N=1024, H=64, M=40, norm=True,
                            window='hanning'):
        X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N,
                         window=window)
        Fs_feature = Fs / H
        phase = np.angle(X) / (2 * np.pi)
        phase_diff = self.__principal_argument(np.diff(phase, axis=1))
        phase_diff2 = self.__principal_argument(np.diff(phase_diff, axis=1))
        novelty_phase = np.sum(np.abs(phase_diff2), axis=0)
        novelty_phase = np.concatenate((novelty_phase, np.array([0, 0])))
        if M > 0:
            local_average = self.__compute_local_average(novelty_phase, M)
            novelty_phase = novelty_phase - local_average
            novelty_phase[novelty_phase < 0] = 0
        if norm:
            max_value = np.max(novelty_phase)
            if max_value > 0:
                novelty_phase = novelty_phase / max_value
        return novelty_phase, Fs_feature
