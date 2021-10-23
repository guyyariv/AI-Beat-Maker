"""
Inputs
"""
import time
import random
import numpy as np
import pydub
import soundfile as sf
import matplotlib.pyplot as plt
import librosa
import librosa.display
# import aubio
from pydub.utils import make_chunks



"""
This function will plot desired graphs
"""


def plot_graphs(audio, samp_rate, times, onset_env, onset_fr, sign_changes):
    fig, axs = plt.subplots(2)
    axs[0].plot(audio)
    axs[0].set_title("audio")
    axs[1].plot(times * samp_rate, onset_env)
    axs[1].vlines(times[onset_fr] * samp_rate, 0, onset_env.max(), 'r', linestyles='--')
    axs[1].set_title("env")
    plt.show()

    S = np.abs(librosa.stft(audio))
    fig, ax = plt.subplots()
    img = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time', ax=ax)
    ax.set_title('Power spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.show()

    plt.plot(times, onset_env)
    plt.vlines(times[onset_fr], 0, onset_env.max(), 'r', linestyles='--')
    plt.show()

    plt.plot(times, onset_env)
    plt.vlines(times[np.argwhere(sign_changes > 0)], 0, onset_env.max(), 'r', linestyles='--')
    plt.show()

    Xdb = librosa.amplitude_to_db(abs(librosa.stft(audio)))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=samp_rate, x_axis='time', y_axis='hz')
    #If to pring log of frequencies
    librosa.display.specshow(Xdb, sr=samp_rate, x_axis='time', y_axis='log')
    plt.colorbar()
    plt.show()


    # hop_length = 512
    fig, ax = plt.subplots(nrows=2, sharex=True)
    # times = librosa.times_like(onset_envelope, sr=sr, hop_length=hop_length)
    M = librosa.feature.melspectrogram(y=audio, sr=samp_rate)
    librosa.display.specshow(librosa.power_to_db(M, ref=np.max), y_axis='mel', x_axis='time', ax=ax[0])
    ax[0].label_outer()
    ax[0].set(title='Mel spectrogram')
    ax[1].plot(times, librosa.util.normalize(onset_env), label='Onset strength')
    ax[1].vlines(times[beats], 0, 1, alpha=0.5, color='r', linestyle='--', label='Beats')
    ax[1].legend()
    plt.show()


def get_wav_data(audio_path):
    audio_data, sample_rate = librosa.load(audio_path)
    return audio_data, sample_rate


"""
This function will track the beat and tempo, and will return an array of sliced sections accordingly.
"""

def beat_tracking_and_slice(audio_data, sample_rate):
    tempo, beats = librosa.beat.beat_track(audio_data, sr=sample_rate)
    onset_env = librosa.onset.onset_strength(y=audio_data, sr=sample_rate)
    # onset_frames = librosa.onset.onset_detect(onset_envelope=onset_envelope, sr=sample_rate)
    # X_stft = librosa.stft(audio_data)
    pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sample_rate)
    beats_plp = np.flatnonzero(librosa.util.localmax(pulse))
    times = librosa.times_like(onset_env, sr=sample_rate)

    m_plp = np.mean(onset_env[beats_plp])
    above_avg_plp = beats_plp[onset_env[beats_plp] >= m_plp]

    peaks = [beats[q] for q in range(len(beats)) if q % 4 == 0]
    inner_peaks = []


    between = [num for num in above_avg_plp if num < peaks[0]]
    inner_peaks.append(between)


    for i in range(len(peaks) - 1):
        between = [num for num in above_avg_plp if peaks[i] < num < peaks[i + 1]]
        inner_peaks.append(between)
    between = [num for num in above_avg_plp if peaks[-1] < num]
    inner_peaks.append(between)

    peak_vals = (times[peaks] * sample_rate).astype(int).flatten()
    slices = np.split(audio_data, peak_vals)[1:-1]
    return onset_env, times, peaks, inner_peaks, peak_vals, slices, tempo, beats


"""
This function will arrange the sliced sections randomly.
"""


def random_arrangement(sliced_audio, min_mult, max_mult):
    rearranged = np.array([])

    # shuffle slices
    random.shuffle(sliced_audio)

    # repeat and remix
    rnd_int = np.random.randint(min_mult, max_mult, size=len(sliced_audio))
    for slice, rnd in zip(sliced_audio, rnd_int):
        repeat = np.tile(slice, rnd)
        rearranged = np.append(rearranged, repeat)

    return rearranged


"""
This function will add drums to the given sample.
"""


def drums():
    pass


"""
This function will 
"""


def novelty_detection():
    pass


"""
This function will 
"""


def pitch_and_chord_detecion():
    pass


def shorten_sample(audio_path, start_time_in_s, end_time_in_s, new_name):
    audio_data, sm_rate = librosa.load(audio_path)
    nw_sample = audio_data[(start_time_in_s * sm_rate):(end_time_in_s * sm_rate)]
    sf.write('samples/{}.wav'.format(new_name), nw_sample, sm_rate)


def plot_reg_and_shuff_graphs(envelope, graphs):
    for g in graphs:
        plt.plot(envelope)
        plt.title(g[1])
        plt.vlines(g[0], 0, envelope.max(), 'r', linestyles='--')
        plt.show()


def save_with_clicks(rearanged_slices, samp_rate, shf_peaks, shf_inner_peaks, name):
    clicks = librosa.clicks(frames=shf_peaks, sr=samp_rate, length=len(rearanged_slices)) + librosa.clicks(frames=np.hstack(shf_inner_peaks), sr=samp_rate, length=len(rearanged_slices))
    # clicks = librosa.clicks(frames=shf_peaks, sr=samp_rate, length=len(rearanged_slices))
    # clicks = librosa.clicks(frames=np.hstack(shf_inner_peaks), sr=samp_rate, length=len(rearanged_slices))
    y_beats = rearanged_slices + clicks
    sf.write('results/{}.wav'.format(name), y_beats, samp_rate)



# if __name__ == '__main__':
#     shorten_sample("samples/samp_3.wav", 0, 48, "samp_4")

if __name__ == '__main__':
    track_name = "samp_1"
    # track_name = "samp_4"
    audio_path = "samples/{}.wav".format(track_name)
    audio_data, samp_rate = get_wav_data(audio_path)
    onset_env, times, peaks, inner_peaks, peak_vals, slices, tempo, beats = beat_tracking_and_slice(audio_data, samp_rate)


    # plot_reg_and_shuff_graphs(onset_env, [[peaks, "original peaks"],
    #                                       [np.hstack(inner_peaks), "original inner peaks"],
    #                                       [np.concatenate((np.array(peaks), np.hstack(inner_peaks))), "all peaks"]])

    rearanged_slices = random_arrangement(slices, 1, 3)

    shf_onset_env, shf_times, shf_peaks, shf_inner_peaks, shf_peak_vals, shf_slices, shf_tempo, shf_beats = beat_tracking_and_slice(rearanged_slices, samp_rate)

    # plot_reg_and_shuff_graphs(shf_onset_env, [[shf_peaks, "shuffled peaks"],
    #                                       [np.hstack(shf_inner_peaks), "shuffled inner peaks"],
    #                                       [np.concatenate((np.array(shf_peaks), np.hstack(shf_inner_peaks))), "all shuffled peaks"]])

    save_with_clicks(rearanged_slices, samp_rate, shf_peaks, np.hstack(shf_inner_peaks), "clicks_{}".format(track_name))

    # print("peaks times: ", list(shf_times[peaks]))
    # print("inner peaks times: ", list(shf_times[np.hstack(inner_peaks).astype(int)]))
    # print(tempo)
    # print(times[-1])
    # print(len(shf_times))
    # print(shf_times[-1])
    # print(len(rearanged_slices))
    #
    # sf.write('results/rearanged_9.wav', rearanged_slices, samp_rate)