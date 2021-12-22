import scipy
import scipy.io.wavfile as wav
import scipy.signal as signal
from matplotlib import pyplot as plt
import numpy as np
import IPython.display as ipd
import librosa
from librosa import display
import soundfile as sf
import random
import utils
import os, sys
from beat_tracking.onset_detection import OnsetDetection

sys.path.append('..')


def novelty_detection(audio_data, sr, audio_total_time, H=512):
    chroma = librosa.feature.chroma_cqt(y=audio_data, sr=sr, hop_length=H)
    # mfcc = librosa.feature.mfcc(y=audio_data, sr=sr)
    # tempo = librosa.feature.tempogram(y=audio_data, sr=sr)
    # chroma_stack = librosa.feature.stack_memory(chroma, n_steps=10, delay=3)
    # rec = librosa.segment.recurrence_matrix(chroma_stack, mode='affinity', self=True)
    # rec_smooth = librosa.segment.path_enhance(rec, 51, window='hann', n_filters=7)

    bounds = librosa.segment.agglomerative(chroma, 128)
    bound_times = librosa.frames_to_time(bounds, sr=sr)
    bound_times = np.append(bound_times, audio_total_time)

    # fig, ax = plt.subplots()
    # librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)
    # ax.vlines(bound_times, 0, chroma.shape[0], color='linen', linestyle='--',
    #           linewidth=2, alpha=0.9, label='Segment boundaries')
    # ax.legend()
    # ax.set(title='Power spectrogram')
    # plt.show()
    return bound_times


def slice_by_novelty_detection(audio_data, sr, audio_total_time, k=16, H=512):
    bound_times = novelty_detection(audio_data, sr, audio_total_time, H=H)
    return find_k_main_slices(bound_times, k)


def find_k_main_slices(bound_times, k):
    k_main = list()
    for i in range(len(bound_times) - 1):
        k_main.append((bound_times[i + 1] - bound_times[i], i, bound_times[i], bound_times[i + 1]))
    k_main_sorted = sorted(k_main, reverse=True)
    slices = np.array(k_main_sorted)[:k][:, 2:]
    return slices

