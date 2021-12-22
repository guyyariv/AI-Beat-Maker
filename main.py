import librosa
import numpy as np
import random
from matplotlib import pyplot as plt
import utils
from beat_tracking import beat_tracking
from novelty_detection import novelty_detection_tools
from novelty_detection import novelty_detection


def slices_random_arrangement(sliced_audio, initial_slices_num=1, total_time_sec=120):
    """
    This function will arrange the sliced sections randomly.
    """
    rearranged = np.array([])
    if not sliced_audio:
        return sliced_audio
    for i in range(initial_slices_num):
        rearranged = np.append(rearranged, sliced_audio[i])

    # shuffle slices
    random.shuffle(sliced_audio)

    # repeat and remix
    rnd_int = np.random.randint(1, 2, size=max(round(3 * len(sliced_audio) / 4), 1))
    for slice, rnd in zip(sliced_audio, rnd_int):
        repeat = np.tile(slice, rnd)
        rearranged = np.append(rearranged, repeat)

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
    return rearranged



def novelty_detection_other(audio_data, samp_rate, N=2048, H=512):
    # Chroma Feature Sequence
    chromagram = librosa.feature.chroma_stft(y=audio_data, sr=samp_rate, tuning=0, norm=2, hop_length=H, n_fft=N)
    # chromagram = librosa.feature.mfcc(y=audio_data, sr=samp_rate)

    # Chroma Feature Sequence and SSM (10 Hz)
    L, H = 1, 1
    X, Fs_feature = novelty_detection_tools.smooth_downsample_feature_sequence(chromagram, samp_rate,
                                                                               filt_len=L, down_sampling=H)
    X = novelty_detection_tools.normalize_feature_sequence(X, norm='2', threshold=0.001)
    # SSM = novelty_detection_algorithm.compute_sm_dot(X, X)
    tempo_rel_min = 0.66
    tempo_rel_max = 1.5
    num = 5
    shift_set = np.array(range(12))
    tempo_rel_set = novelty_detection_tools.compute_tempo_rel_set(tempo_rel_min=tempo_rel_min, tempo_rel_max=tempo_rel_max, num=num)
    S, I = novelty_detection_tools.compute_sm_ti(X, X, L=L, tempo_rel_set=tempo_rel_set, shift_set=shift_set, direction=2)

    # S = novelty_detection_algorithm.threshold_matrix(S, thresh=[0.2,0.2], strategy='local')
    SSM_norm = novelty_detection_tools.normalization_properties_ssm(S)
    Ls = [40]
    for L in Ls:
        novelty_function = novelty_detection_tools.compute_novelty_ssm(SSM_norm, L=L)
        plt.plot(novelty_function)
        plt.show()
        return novelty_function


if __name__ == "__main__":
    track_name = "John Coltrane A Love Supreme"
    audio_path = "samples/{}.wav".format(track_name)
    audio_data, samp_rate = utils.get_wav_data(audio_path)
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
    estimated_bpm = np.mean(np.array(estimated_bpm))
    print(estimated_bpm)
    tempo_estimated, peaks = beat_tracking.beat_tracking(rearanged_slices, samp_rate)
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
