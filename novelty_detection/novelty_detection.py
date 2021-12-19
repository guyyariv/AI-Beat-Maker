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


def novelty_detection(audio_data, sr, H=512):
    audio_total_time = librosa.get_duration(audio_data)
    chroma = librosa.feature.chroma_cqt(y=audio_data, sr=sr, hop_length=H)
    # mfcc = librosa.feature.mfcc(y=audio_data, sr=sr)
    # tempo = librosa.feature.tempogram(y=audio_data, sr=sr)
    # chroma_stack = librosa.feature.stack_memory(chroma, n_steps=10, delay=3)
    # rec = librosa.segment.recurrence_matrix(chroma_stack, mode='affinity', self=True)
    # rec_smooth = librosa.segment.path_enhance(rec, 51, window='hann', n_filters=7)

    bounds = librosa.segment.agglomerative(chroma, 3)
    bound_times = librosa.frames_to_time(bounds, sr=sr)
    bound_times = np.append(bound_times, audio_total_time)

    fig, ax = plt.subplots()
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)
    ax.vlines(bound_times, 0, chroma.shape[0], color='linen', linestyle='--',
              linewidth=2, alpha=0.9, label='Segment boundaries')
    ax.legend()
    ax.set(title='Power spectrogram')
    plt.show()
    max_distance = 0
    winner_ind = 0
    for i in range(len(bound_times) - 1):
        if (bound_times[i + 1] - bound_times[i]) > max_distance:
            max_distance = bound_times[i + 1] - bound_times[i]
            winner_ind = i
    return bound_times[winner_ind:winner_ind+2]

