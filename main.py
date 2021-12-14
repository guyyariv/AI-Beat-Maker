import numpy as np
import random

import pr
import utils
from beat_tracking import beat_tracking_algorithm


def random_arrangement(sliced_audio, initial_slices_num=4):
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


if __name__ == "__main__":
    track_name = "30_sec_test"
    audio_path = "samples/{}.wav".format(track_name)
    audio_data, samp_rate = utils.get_wav_data(audio_path)
    tempo_estimated, beat_slices = beat_tracking_algorithm.slice_by_beat_tracking(audio_data, samp_rate)
    rearanged_slices = random_arrangement(beat_slices, initial_slices_num=4)
    peaks = beat_tracking_algorithm.plp(rearanged_slices, samp_rate, tempo_min=tempo_estimated-5, tempo_max=tempo_estimated+5)
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
