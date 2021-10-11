"""
Inputs
"""
import random
import numpy as np
import pydub
import soundfile as sf
import matplotlib.pyplot as plt
import librosa
import librosa.display

# import magenta


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
    tempo, beats = librosa.beat.beat_track(audio_data, sr=sample_rate)
    return audio_data, sample_rate, tempo, beats


"""
This function will track the beat and tempo, and will return an array of sliced sections accordingly.
"""


def beat_tracking_and_slice(audio_data, sample_rate):
    onset_envelope = librosa.onset.onset_strength(audio_data, sr=sample_rate)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_envelope, sr=sample_rate)
    times = librosa.times_like(onset_envelope, sr=sample_rate)
    X = librosa.stft(audio_data)
    Xdb = librosa.amplitude_to_db(abs(X))

    col_sums = Xdb.sum(axis=0)
    col_sign = np.sign(col_sums)
    sign_changes = ((np.roll(col_sign, 1) - col_sign) < 0).astype(int)

    peaks = (times[np.argwhere(sign_changes > 0)] * sample_rate).astype(int).flatten()
    return np.split(audio_data, peaks), sign_changes, times, onset_frames, onset_envelope


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


if __name__ == '__main__':
    audio_path = "resources_test.wav"
    audio_data, sample_rate, tempo, beats = get_wav_data(audio_path)
    slices, sign_times, time, ons_fr, ons_env = beat_tracking_and_slice(audio_data, sample_rate)
    min_rep = 1
    max_rep = 3
    rem = random_arrangement(slices, min_rep, max_rep)
    # plot_graphs(audio_data, sample_rate, time, ons_env, ons_fr, sign_times)
    # sf.write('test_one_pr_file.wav', rem, sample_rate)
