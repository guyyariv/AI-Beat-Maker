import librosa
import numpy as np
import random
from matplotlib import pyplot as plt
import pr
import utils
from beat_tracking import beat_tracking_algorithm
from novelty_detection import novelty_detection_algorithm


def random_arrangement(sliced_audio, initial_slices_num=4, total_time_sec=120, tempo_estimated=120):
    """
    This function will arrange the sliced sections randomly.
    """
    tempo_estimated = tempo_estimated / 60
    total_slices = round(total_time_sec / tempo_estimated)
    rearranged = np.array([])
    for i in range(initial_slices_num):
        rearranged = np.append(rearranged, sliced_audio[i])

    # shuffle slices
    random.shuffle(sliced_audio)

    # repeat and remix
    rnd_int = np.random.randint(1, round((total_slices / len(sliced_audio)) * 2), size=len(sliced_audio))
    for slice, rnd in zip(sliced_audio, rnd_int):
        repeat = np.tile(slice, rnd)
        rearranged = np.append(rearranged, repeat)

    return rearranged


def random_arrangement_other(sliced_audio, initial_slices_num=4):
    """
    This function will arrange the sliced sections randomly.
    """
    rearranged = np.array([])
    initial_slices = np.array([])
    if len(sliced_audio) > initial_slices_num:
        initial_slices = sliced_audio[:initial_slices_num]
    # shuffle slices
    # random.shuffle(sliced_audio)
    # repeat and remix
    rnd_int = np.random.randint(initial_slices_num, len(sliced_audio), size=initial_slices_num)
    for slice in initial_slices:
        rearranged = np.append(rearranged, slice)
    for _ in range(np.min(rnd_int)):
        if _ % 2 == 0:
            for slice in sliced_audio[np.min(rnd_int):np.max(rnd_int)]:
                rearranged = np.append(rearranged, slice)
        else:
            for slice in sliced_audio[np.max(rnd_int):]:
                rearranged = np.append(rearranged, slice)
    return rearranged


def novelty_detection(audio_data, samp_rate, N=4096, H=1024):
    # Chroma Feature Sequence
    chromagram = librosa.feature.chroma_stft(y=audio_data, sr=samp_rate, tuning=0, norm=2, hop_length=H, n_fft=N)

    # Chroma Feature Sequence and SSM (10 Hz)
    L, H = 1, 1
    X, Fs_feature = novelty_detection_algorithm.smooth_downsample_feature_sequence(chromagram, samp_rate,
                                                                 filt_len=L, down_sampling=H)
    X = novelty_detection_algorithm.normalize_feature_sequence(X, norm='2', threshold=0.001)
    # SSM = novelty_detection_algorithm.compute_sm_dot(X, X)
    tempo_rel_min = 0.66
    tempo_rel_max = 1.5
    num = 5
    shift_set = np.array(range(12))
    tempo_rel_set = novelty_detection_algorithm.compute_tempo_rel_set(tempo_rel_min=tempo_rel_min, tempo_rel_max=tempo_rel_max, num=num)
    S, I = novelty_detection_algorithm.compute_sm_ti(X, X, L=L, tempo_rel_set=tempo_rel_set, shift_set=shift_set, direction=2)

    # S = novelty_detection_algorithm.threshold_matrix(S, thresh=[0.2,0.2], strategy='local')
    SSM_norm = novelty_detection_algorithm.normalization_properties_ssm(S)
    Ls = [40]
    for L in Ls:
        novelty_function = novelty_detection_algorithm.compute_novelty_ssm(SSM_norm, L=L)
        plt.plot(novelty_function)
        plt.show()


if __name__ == "__main__":
    track_name = "30_sec_test"
    audio_path = "samples/{}.wav".format(track_name)
    audio_data, samp_rate = utils.get_wav_data(audio_path)
    novelty_detection(audio_data, samp_rate)
    tempo_estimated, beat_slices = beat_tracking_algorithm.slice_by_beat_tracking(audio_data, samp_rate)
    print(tempo_estimated)
    rearanged_slices = random_arrangement(beat_slices, total_time_sec=120, tempo_estimated=tempo_estimated)
    tempo_estimated, peaks = beat_tracking_algorithm.beat_tracking(rearanged_slices, samp_rate, start_bpm=tempo_estimated)
    peaks_times = list()
    inner_peaks_times = list()
    for i in range(len(peaks)):
        if i % 4 == 0:
            peaks_times.append(peaks[i])
        else:
            inner_peaks_times.append(peaks[i])
    print("peaks_times = ", peaks_times)
    print("inner_peaks_times = ", inner_peaks_times)

    utils.add_clicks_and_save(rearanged_slices, samp_rate, peaks, "{}".format(track_name), clicks=False)
    utils.add_clicks_and_save(rearanged_slices, samp_rate, peaks, "clicks_{}".format(track_name), clicks=True)
