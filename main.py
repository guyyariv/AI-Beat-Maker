import librosa
import numpy as np
import random
from matplotlib import pyplot as plt
import utils
from beat_tracking import beat_tracking
from novelty_detection import novelty_detection_tools
from novelty_detection import novelty_detection
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


def interval_arrangement(slice, sr, frame_length=2):
    _, beat_frames = librosa.beat.beat_track(y=slice, sr=sr)
    beat_samples = librosa.frames_to_samples(beat_frames)
    # if len(beat_samples) < frame_length:
    #     frame_length = int(len(beat_samples) / 2)
    inner_intervals = librosa.util.frame(beat_samples, frame_length=frame_length, hop_length=1).T
    onset_env = librosa.onset.onset_strength(slice, sr=sr,
                                             aggregate=np.median)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env,
                                           sr=sr)
    hop_length = 512
    fig, ax = plt.subplots(nrows=2, sharex=True)
    times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)
    M = librosa.feature.melspectrogram(y=slice, sr=sr, hop_length=hop_length)
    librosa.display.specshow(librosa.power_to_db(M, ref=np.max),
                             y_axis='mel', x_axis='time', hop_length=hop_length,
                             ax=ax[0])
    ax[0].label_outer()
    ax[0].set(title='Mel spectrogram')
    ax[1].plot(times, librosa.util.normalize(onset_env),
               label='Onset strength')
    ax[1].vlines(times[beats], 0, 1, alpha=0.5, color='r',
                 linestyle='--', label='Beats')
    ax[1].legend()
    plt.show()
    return librosa.effects.remix(slice, inner_intervals[::-1], align_zeros=False)


def all_process(track_name, audio_path):
    audio_data, samp_rate = utils.get_wav_data(audio_path)
    audio_data, index = librosa.effects.trim(audio_data)
    audio_total_time = librosa.get_duration(audio_data)
    novelty_slices = novelty_detection.slice_by_novelty_detection(audio_data, samp_rate, audio_total_time)
    slices = list()
    estimated_bpm = [120]
    for slice in novelty_slices:
        time_1, time_2 = (slice * samp_rate).astype(np.int32)
        tempo, bt_slices = beat_tracking.slice_by_beat_tracking(audio_data[time_1:time_2], samp_rate, start_bpm=estimated_bpm[-1])
        slices_ran = slices_random_arrangement(bt_slices, total_time_sec=audio_total_time/len(novelty_slices))
        slices.append(slices_ran)
        estimated_bpm.append(tempo)

    rearanged_slices = random_arrangement(slices)
    print(estimated_bpm)
    estimated_bpm = np.argmax(np.bincount(estimated_bpm))
    print(estimated_bpm)
    tempo_estimated, peaks = beat_tracking.beat_tracking(rearanged_slices, samp_rate, start_bpm=estimated_bpm)
    inner_peaks = beat_tracking.plp(rearanged_slices, samp_rate)
    peaks_frame = peaks
    print("peaks_times = ", peaks_frame)
    print("all_peaks_from_beat_tracking = ", peaks)
    print("inner_peaks_times = ", inner_peaks)

    utils.add_clicks_and_save(rearanged_slices, samp_rate, peaks_frame, inner_peaks, "{}".format(track_name), clicks=False)
    utils.add_clicks_and_save(rearanged_slices, samp_rate, peaks_frame, inner_peaks, "clicks_{}".format(track_name), clicks=True)

    return rearanged_slices, peaks_frame, peaks, inner_peaks


def remix(audio_path):
    audio_data, samp_rate = utils.get_wav_data(audio_path)
    audio_data, index = librosa.effects.trim(audio_data)
    tempo, beat_frames = librosa.beat.beat_track(y=audio_data, sr=samp_rate, hop_length=512)
    beat_samples = librosa.frames_to_samples(beat_frames)
    intervals = librosa.util.frame(beat_samples, frame_length=2, hop_length=1).T
    y_out = librosa.effects.remix(audio_data, intervals[::-1])
    return tempo, y_out, samp_rate


if __name__ == "__main__":
    # utils.shorten_sample('samples/Infected Mushroom - Becoming Insane.wav', 0, 25, 'Becoming_Insane')
    track_name = "new_gipsy kings"
    audio_path = "samples/{}.wav".format(track_name)
    audio_data, samp_rate = utils.get_wav_data(audio_path)
    audio_data, index = librosa.effects.trim(audio_data)
    k = 8
    intervals, bound_segs = novelty_detection.novelty_detection(audio_data, samp_rate, k)
    output = list()
    # res = zip(bound_segs, intervals)
    # res = list(res)
    # res = sorted(res, key=lambda x: x[0])
    for interval in intervals:
        try:
            inner_arrangement = interval_arrangement(audio_data[np.int32(interval[0] * samp_rate):np.int32(interval[1] * samp_rate)], samp_rate)
        except:
            inner_arrangement = np.array([])
        output.append(inner_arrangement)
    output = random_arrangement(output)
    sf.write('results/new_{}.wav'.format(track_name), output, samp_rate)
