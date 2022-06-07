import matplotlib.pyplot as plt
import librosa
import numpy as np
import utils
from algorithms import beat_tracking
from algorithms.chord_recognition import rearrange_by_chord_recognition
from algorithms.novelty_detection import NoveltyDetection
import soundfile as sf



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


def mae_calculator(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions))


def beat_maker(track_name):

    audio_path = "samples/{}.wav".format(track_name)
    audio_data, samp_rate = utils.get_wav_data(audio_path)
    audio_data, index = librosa.effects.trim(audio_data)
    k = 2
    intervals, bound_segs = NoveltyDetection(audio_data, samp_rate, k).novelty_detection()

    output = list()
    for interval in intervals:
        try:
            time_1, time_2 = np.array([(interval[0] * samp_rate).astype(np.int32), (interval[1] * samp_rate).astype(np.int32)])
            tempo, bt_slices = beat_tracking.slice_by_beat_tracking(audio_data[time_1:time_2],
                                                                    samp_rate)
            inner_arrangement = rearrange_by_chord_recognition(bt_slices, samp_rate)
        except:
            continue
        output.append(inner_arrangement)
    output = rearrange_by_chord_recognition(output, samp_rate)
    tempo, beats = librosa.beat.beat_track(y=output, sr=samp_rate)
    print(tempo)
    #
    # predict.generate(tempo=tempo, length=2000)
    #

    s2_wav_data, _ = utils.get_wav_data('drums_results/drums.wav')
    s2_wav_data = librosa.effects.trim(librosa.util.normalize(s2_wav_data))[0] * 2

    s1_wav_data = librosa.util.normalize(output)

    tempo_2, beats_2 = librosa.beat.beat_track(y=s2_wav_data, sr=samp_rate)

    s1_wav_len = s1_wav_data.shape[0]
    s2_wav_len = s2_wav_data.shape[0]
    min_length = min(s1_wav_len, s2_wav_len)

    s3_wav_data = s1_wav_data[:min_length] + s2_wav_data[:min_length]

    length = min(len(beats), len(beats_2))

    return mae_calculator(beats[:length], beats_2[:length]), s3_wav_data, samp_rate
    # return 2, output, samp_rate


if __name__ == "__main__":

    track_name = "drumless"
    mae, s3_wav_data, samp_rate = beat_maker(track_name)
    print(mae, 0)

    min_mae = mae
    final_res = s3_wav_data
    idx = 0

    while mae > 30 and idx < 5:

        idx += 1
        mae, s3_wav_data, samp_rate = beat_maker(track_name)
        if mae < min_mae:
            min_mae = mae
            final_res = s3_wav_data
        print(mae, idx)

    sf.write(f'results/full_process_{track_name.replace(" ", "_").lower()}.wav', final_res, samp_rate)
