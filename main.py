import librosa
import numpy as np
import utils
from beat_tracking import beat_tracking
from novelty_detection import novelty_detection
import soundfile as sf
from rearrangement import random_rearrangement


def all_process(track_name, audio_path):
    audio_data, samp_rate = utils.get_wav_data(audio_path)
    audio_data, index = librosa.effects.trim(audio_data)
    audio_total_time = librosa.get_duration(audio_data)
    novelty_slices = novelty_detection.slice_by_novelty_detection(audio_data, samp_rate, audio_total_time)
    slices = list()
    estimated_bpm = [120]
    for slice in novelty_slices:
        time_1, time_2 = (slice * samp_rate).astype(np.int32)
        tempo, bt_slices = beat_tracking.slice_by_beat_tracking(audio_data[time_1:time_2],
                                                                samp_rate, start_bpm=estimated_bpm[-1])
        slices_ran = random_rearrangement.slices_random_arrangement(bt_slices,
                                                                    total_time_sec=audio_total_time/len(novelty_slices))
        slices.append(slices_ran)
        estimated_bpm.append(tempo)

    rearanged_slices = random_rearrangement.random_arrangement(slices)
    print(estimated_bpm)
    estimated_bpm = np.argmax(np.bincount(estimated_bpm))
    print(estimated_bpm)
    tempo_estimated, peaks = beat_tracking.beat_tracking(rearanged_slices, samp_rate, start_bpm=estimated_bpm)
    inner_peaks = beat_tracking.plp(rearanged_slices, samp_rate)
    peaks_frame = peaks
    print("peaks_times = ", peaks_frame)
    print("all_peaks_from_beat_tracking = ", peaks)
    print("inner_peaks_times = ", inner_peaks)

    utils.add_clicks_and_save(rearanged_slices, samp_rate, peaks_frame, inner_peaks,
                              "{}".format(track_name), clicks=False)
    utils.add_clicks_and_save(rearanged_slices, samp_rate, peaks_frame, inner_peaks,
                              "clicks_{}".format(track_name), clicks=True)

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
    track_name = "beach"
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
            inner_arrangement = random_rearrangement.interval_arrangement(
                audio_data[np.int32(interval[0] * samp_rate):np.int32(interval[1] * samp_rate)], samp_rate)
        except:
            inner_arrangement = np.array([])
        output.append(inner_arrangement)
    output = random_rearrangement.random_arrangement(output)
    sf.write('results/new_{}.wav'.format(track_name), output, samp_rate)
