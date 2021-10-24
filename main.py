import scipy
import scipy.io.wavfile as wav
import scipy.signal as signal
from matplotlib import pyplot as plt
import numpy as np
import IPython.display as ipd
import librosa
import soundfile as sf
import random
import utils
import os, sys
from beat_tracking.onset_detection import OnsetDetection

sys.path.append('..')


def beat_tracking_and_slice(audio_data, sample_rate):
    # spectral based onset envelop
    onset_env = librosa.onset.onset_strength(y=audio_data, sr=sample_rate)
    #
    times = librosa.times_like(onset_env, sr=sample_rate)
    # onset_frames = librosa.onset.onset_detect(onset_envelope=onset_envelope, sr=sample_rate)
    # X_stft = librosa.stft(audio_data)

    # estimate tempo from onset correlation
    # pick peaks in onset strength approximately consistent with estimated
    # tempo
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env)
    # find a locally stable tempo for each frame
    pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sample_rate)
    beats_plp = np.flatnonzero(librosa.util.localmax(pulse))

    m_plp = np.mean(onset_env[beats_plp])
    above_avg_plp = beats_plp[onset_env[beats_plp] >= m_plp]

    peaks = beats[::4]
    inner_peaks = np.delete(above_avg_plp, np.nonzero(np.in1d(above_avg_plp,
                                                              peaks))[0])

    peak_vals = (times[peaks] * sample_rate).astype(int)
    slices = np.split(audio_data, peak_vals)[1:-1]
    return onset_env, times, peaks, inner_peaks, peak_vals, slices, tempo, beats


def random_arrangement(sliced_audio, min_mult, max_mult):
    """
    This function will arrange the sliced sections randomly.
    """
    rearranged = np.array([])

    # shuffle slices
    random.shuffle(sliced_audio)

    # repeat and remix
    rnd_int = np.random.randint(min_mult, max_mult, size=len(sliced_audio))
    for slice, rnd in zip(sliced_audio, rnd_int):
        repeat = np.tile(slice, rnd)
        rearranged = np.append(rearranged, repeat)

    return rearranged


if __name__ == '__main__':
    track_name = "Maarten Schellekens - Mallet Play"
    audio_path = "samples/{}.wav".format(track_name)
    audio_data, samp_rate = utils.get_wav_data(audio_path)
    onset_env, times, peaks, inner_peaks, peak_vals, slices, tempo, beats = beat_tracking_and_slice(
        audio_data, samp_rate)
    utils.plot_reg_and_shuff_graphs(onset_env, [[peaks, "original peaks"],
                                                [np.hstack(inner_peaks),
                                                 "original inner peaks"],
                                                [np.concatenate((np.array(
                                                    peaks), np.hstack(
                                                    inner_peaks))),
                                                 "all peaks"]])
    rearanged_slices = random_arrangement(slices, 1, 3)
    shf_onset_env, shf_times, shf_peaks, shf_inner_peaks, shf_peak_vals, shf_slices, shf_tempo, shf_beats = beat_tracking_and_slice(
        rearanged_slices, samp_rate)

    utils.plot_reg_and_shuff_graphs(shf_onset_env, [[shf_peaks, "shuffled "
                                                                "peaks"],
                                                    [np.hstack(
                                                        shf_inner_peaks),
                                                     "shuffled inner peaks"],
                                                    [np.concatenate((np.array(
                                                        shf_peaks), np.hstack(
                                                        shf_inner_peaks))),
                                                     "all shuffled peaks"]])

    utils.save_with_clicks(rearanged_slices, samp_rate, shf_peaks,
                           np.hstack(shf_inner_peaks),
                           "clicks_{}".format(track_name))

    print("peaks times: ", list(shf_times[peaks]))
    print("inner peaks times: ",
          list(shf_times[np.hstack(inner_peaks).astype(int)]))
    print(tempo)
    print(times[-1])
    print(len(shf_times))
    print(shf_times[-1])
    print(len(rearanged_slices))

    sf.write('results/rearanged_9.wav', rearanged_slices, samp_rate)
