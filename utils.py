import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
import soundfile as sf
import librosa


def create_wav_file_from_mp3_file(src, dst):
    """
    :param file_name: mp3 file
    :param dst: destination wav file name
    """
    sound = AudioSegment.from_mp3(src)
    sound.export(dst, format="wav")


def get_wav_data(audio_path):
    """
    This function will track the beat and tempo, and will return an array of sliced sections accordingly.
    """
    audio_data, sample_rate = librosa.load(audio_path)
    return audio_data, sample_rate


def plot_beat_tracking_graphs(audio, samp_rate, times, onset_env, onset_fr, sign_changes,
                              beats):
    """
    This function will plot desired graphs
    """
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


def save_with_clicks(rearanged_slices, samp_rate, shf_peaks, shf_inner_peaks, name):
    clicks = librosa.clicks(frames=shf_peaks, sr=samp_rate, length=len(rearanged_slices)) + librosa.clicks(frames=np.hstack(shf_inner_peaks), sr=samp_rate, length=len(rearanged_slices))
    # clicks = librosa.clicks(frames=shf_peaks, sr=samp_rate, length=len(rearanged_slices))
    # clicks = librosa.clicks(frames=np.hstack(shf_inner_peaks), sr=samp_rate, length=len(rearanged_slices))
    y_beats = rearanged_slices + clicks
    sf.write('results/{}.wav'.format(name), y_beats, samp_rate)


def add_clicks_and_save(audio, samp_rate, peaks, name, clicks=True):
    click_times = np.zeros((len(audio),))
    if clicks:
        click_times = librosa.clicks(times=peaks, sr=samp_rate, length=len(audio))
    y_beats = audio + click_times
    sf.write('results/{}.wav'.format(name), y_beats, samp_rate)


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