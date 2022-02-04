import random
from matplotlib import pyplot as plt
import librosa
import numpy as np


def slices_random_arrangement(sliced_audio, total_time_sec=None):
    """
    This function will arrange the sliced sections randomly.
    """
    rearranged = np.array([])
    if not sliced_audio:
        return sliced_audio
    # for i in range(initial_slices_num):
    #     rearranged = np.append(rearranged, sliced_audio[i])

    # shuffle slices
    random.shuffle(sliced_audio)

    # repeat and remix
    rnd_int = np.random.randint(1, 2, size=max(round(3 * len(sliced_audio) / 4), 1))
    for slice, rnd in zip(sliced_audio, rnd_int):
        # repeat = np.tile(slice, rnd)
        rearranged = np.append(rearranged, slice)
    return rearranged


def random_arrangement(slices):
    """
    :param slices: list of np.arrays
    :return: 1-d np.array
    """
    random.shuffle(slices)
    rearranged = np.array([])
    for slice in slices:
        rearranged = np.append(rearranged, slice)
        # rearranged = np.append(rearranged, np.random.choice(slices))
    return rearranged


def interval_arrangement(slice, sr, frame_length=8):
    _, beat_frames = librosa.beat.beat_track(y=slice, sr=sr)
    beat_samples = librosa.frames_to_samples(beat_frames)

    inner_intervals = librosa.util.frame(beat_samples, frame_length=frame_length, hop_length=1).T
    onset_env = librosa.onset.onset_strength(slice, sr=sr, aggregate=np.median)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    hop_length = 512
    fig, ax = plt.subplots(nrows=2, sharex=True)
    times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)
    M = librosa.feature.melspectrogram(y=slice, sr=sr, hop_length=hop_length)
    librosa.display.specshow(librosa.power_to_db(M, ref=np.max), y_axis='mel', x_axis='time',
                             hop_length=hop_length, ax=ax[0])
    ax[0].label_outer()
    ax[0].set(title='Mel spectrogram')
    ax[1].plot(times, librosa.util.normalize(onset_env), label='Onset strength')
    ax[1].vlines(times[beats], 0, 1, alpha=0.5, color='r', linestyle='--', label='Beats')
    ax[1].legend()
    plt.show()
    return librosa.effects.remix(slice, inner_intervals[::-1], align_zeros=False)