import random
import matplotlib.pyplot as plt
import librosa
import numpy as np
import utils
from algorithms import beat_tracking, novelty_detection
from algorithms.novelty_detection import NoveltyDetection
import soundfile as sf


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


def interval_arrangement(slice, sr, frame_length=2, show=False):
    _, beat_frames = librosa.beat.beat_track(y=slice, sr=sr)
    beat_samples = librosa.frames_to_samples(beat_frames)

    inner_intervals = librosa.util.frame(beat_samples, frame_length=frame_length, hop_length=1).T
    onset_env = librosa.onset.onset_strength(slice, sr=sr, aggregate=np.median)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

    if show:
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


def remix(audio_path):
    audio_data, samp_rate = utils.get_wav_data(audio_path)
    audio_data, index = librosa.effects.trim(audio_data)
    tempo, beat_frames = librosa.beat.beat_track(y=audio_data, sr=samp_rate, hop_length=512)
    beat_samples = librosa.frames_to_samples(beat_frames)
    intervals = librosa.util.frame(beat_samples, frame_length=2, hop_length=1).T
    y_out = librosa.effects.remix(audio_data, intervals[::-1])
    return tempo, y_out, samp_rate


if __name__ == "__main__":
    track_name = "drumless"
    audio_path = "samples/{}.wav".format(track_name)
    audio_data, samp_rate = utils.get_wav_data(audio_path)
    audio_data, index = librosa.effects.trim(audio_data)
    k = 4
    intervals, bound_segs = NoveltyDetection(audio_data, samp_rate, k).novelty_detection()

    output = list()
    # res = zip(bound_segs, intervals)
    # res = list(res)
    # res = sorted(res, key=lambda x: x[0])
    for interval in intervals:
        try:
            # inner_arrangement = interval_arrangement(
            #     audio_data[np.int32(interval[0] * samp_rate):np.int32(interval[1] * samp_rate)], samp_rate)
            time_1, time_2 = np.array([(interval[0] * samp_rate).astype(np.int32), (interval[1] * samp_rate).astype(np.int32)])
            tempo, bt_slices = beat_tracking.slice_by_beat_tracking(audio_data[time_1:time_2],
                                                                    samp_rate)
            inner_arrangement = slices_random_arrangement(bt_slices)
        except:
            inner_arrangement = np.array([])
        output.append(inner_arrangement)
    output = random_arrangement(output)
    tempo, beats = librosa.beat.beat_track(y=output, sr=samp_rate)

    # predict.generate(tempo=tempo, length=2000)

    s1_wav_data = output
    s2_wav_data, _ = utils.get_wav_data('output2.wav')
    s2_wav_data = librosa.effects.trim(s2_wav_data)[0] * 8

    s1_wav_len = s1_wav_data.shape[0]
    s2_wav_len = s2_wav_data.shape[0]
    min_length = min(s1_wav_len, s2_wav_len)

    s3_wav_data = s1_wav_data[:min_length] + s2_wav_data[:min_length]

    sf.write(f'results/_new_{track_name.replace(" ", "_").lower()}.wav', s3_wav_data, samp_rate)

