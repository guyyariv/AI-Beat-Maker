from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
import librosa
from librosa import display
from scipy import ndimage
import scipy
import matplotlib.pyplot as plt
import sklearn.cluster
import os, sys
sys.path.append('..')


def basic_novelty_detection(audio_data, sr, audio_total_time, H=1024):
    # mfcc = librosa.feature.mfcc(y=audio_data, sr=sr)
    # tempo = librosa.feature.tempogram(y=audio_data, sr=sr)
    y_harmonic = librosa.effects.harmonic(audio_data)
    chroma = librosa.feature.chroma_stft(y=y_harmonic, sr=sr, hop_length=H)
    bounds = librosa.segment.agglomerative(chroma, 16)
    bound_times = librosa.frames_to_time(bounds, sr=sr)
    bound_times = np.append(bound_times, audio_total_time)

    fig, ax = plt.subplots()
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)
    ax.vlines(bound_times, 0, chroma.shape[0], color='linen', linestyle='--',
              linewidth=2, alpha=0.9, label='Segment boundaries')
    ax.legend()
    ax.set(title='Power spectrogram')
    plt.show()
    return bound_times


def slice_by_novelty_detection(audio_data, sr, audio_total_time, k=4, H=1024):
    bound_times = basic_novelty_detection(audio_data, sr, audio_total_time)
    return find_k_main_slices(bound_times, k)


def find_k_main_slices(bound_times, k):
    k_main = list()
    for i in range(len(bound_times) - 1):
        k_main.append((bound_times[i + 1] - bound_times[i], i, bound_times[i], bound_times[i + 1]))
    k_main_sorted = sorted(k_main, reverse=True)
    slices = np.array(k_main_sorted)[::2][:, 2:]
    return slices


class NoveltyDetection:
    def __init__(self, audio_data, sr, k):
        self.audio_data = audio_data
        self.sr = sr
        self.k = k
        self.BINS_PER_OCTAVE = 12 * 3
        self.N_OCTAVES = 7
        self.colors = plt.get_cmap('Paired', self.k)

    def compute_log_power_cqt(self, show=False):
        C = librosa.amplitude_to_db(np.abs(librosa.cqt(y=self.audio_data, sr=self.sr, bins_per_octave=self.BINS_PER_OCTAVE,
                                                       n_bins=self.N_OCTAVES * self.BINS_PER_OCTAVE)), ref=np.max)
        if show:
            fig, ax = plt.subplots()
            librosa.display.specshow(C, y_axis='cqt_hz', sr=self.sr, bins_per_octave=self.BINS_PER_OCTAVE, x_axis='time', ax=ax)
            plt.show()
        return C

    def reduce_cqt_dimensionality(self, C, show):
        tempo, beats = librosa.beat.beat_track(y=self.audio_data, sr=self.sr, trim=False)
        Csync = librosa.util.sync(C, beats, aggregate=np.median)

        # For plotting purposes, we'll need the timing of the beats
        # we fix_frames to include non-beat frames 0 and C.shape[1] (final frame)
        beat_times = librosa.frames_to_time(librosa.util.fix_frames(beats, x_min=0), sr=self.sr)
        if show:
            fig, ax = plt.subplots()
            librosa.display.specshow(Csync, bins_per_octave=12*3,
                                     y_axis='cqt_hz', x_axis='time',
                                     x_coords=beat_times, ax=ax)
        return Csync, beats, beat_times

    def build_weighted_recurrence_matrix(self, Csync):
        R = librosa.segment.recurrence_matrix(Csync, width=3, mode='affinity', sym=True)
        df = librosa.segment.timelag_filter(ndimage.median_filter)
        Rf = df(R, size=(1, 7))
        return Rf

    def build_sequence_matrix(self, beats, Rf, beat_times, show):
        mfcc = librosa.feature.mfcc(y=self.audio_data, sr=self.sr)
        Msync = librosa.util.sync(mfcc, beats)
        path_distance = np.sum(np.diff(Msync, axis=1)**2, axis=0)
        sigma = np.median(path_distance)
        path_sim = np.exp(-path_distance / sigma)
        R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)
        deg_path = np.sum(R_path, axis=1)
        deg_rec = np.sum(Rf, axis=1)
        mu = deg_path.dot(deg_path + deg_rec) / np.sum((deg_path + deg_rec)**2)
        A = mu * Rf + (1 - mu) * R_path

        if show:
            fig, ax = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(10, 4))
            librosa.display.specshow(Rf, cmap='inferno_r', y_axis='time', x_axis='s',
                                     y_coords=beat_times, x_coords=beat_times, ax=ax[0])
            ax[0].set(title='Recurrence similarity')
            ax[0].label_outer()
            librosa.display.specshow(R_path, cmap='inferno_r', y_axis='time', x_axis='s',
                                     y_coords=beat_times, x_coords=beat_times, ax=ax[1])
            ax[1].set(title='Path similarity')
            ax[1].label_outer()
            librosa.display.specshow(A, cmap='inferno_r', y_axis='time', x_axis='s',
                                     y_coords=beat_times, x_coords=beat_times, ax=ax[2])
            ax[2].set(title='Combined graph')
            ax[2].label_outer()
            plt.show()
        return A

    def clustering(self, X, beat_times, Rf, show):
        KM = sklearn.cluster.KMeans(n_clusters=self.k)
        seg_ids = KM.fit_predict(X)
        if show:
            fig, ax = plt.subplots(ncols=3, sharey=True, figsize=(10, 4))
            librosa.display.specshow(Rf, cmap='inferno_r', y_axis='time', y_coords=beat_times, ax=ax[1])
            ax[1].set(title='Recurrence matrix')
            ax[1].label_outer()
            librosa.display.specshow(X, y_axis='time', y_coords=beat_times, ax=ax[0])
            ax[0].set(title='Structure components')
            img = librosa.display.specshow(np.atleast_2d(seg_ids).T, cmap=self.colors, y_axis='time',
                                           y_coords=beat_times, ax=ax[2])
            ax[2].set(title='Estimated segments')
            ax[2].label_outer()
            fig.colorbar(img, ax=[ax[2]], ticks=range(self.k))
            plt.show()
        return seg_ids

    def novelty_detection(self, show=False):
        C = self.compute_log_power_cqt(show)
        Csync, beats, beat_times = self.reduce_cqt_dimensionality(C, show)
        Rf = self.build_weighted_recurrence_matrix(Csync)
        A = self.build_sequence_matrix(beats, Rf, beat_times, show)

        L = scipy.sparse.csgraph.laplacian(A, normed=True)
        # and its spectral decomposition
        evals, evecs = scipy.linalg.eigh(L)
        # We can clean this up further with a median filter.
        # This can help smooth over small discontinuities
        evecs = scipy.ndimage.median_filter(evecs, size=(9, 1))
        # cumulative normalization is needed for symmetric normalize laplacian eigenvectors
        Cnorm = np.cumsum(evecs**2, axis=1)**0.5
        # If we want k clusters, use the first k normalized eigenvectors.
        X = evecs[:, :self.k] / Cnorm[:, self.k-1:self.k]

        # Plot the resulting representation
        if show:
            fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(10, 5))
            librosa.display.specshow(Rf, cmap='inferno_r', y_axis='time', x_axis='time', y_coords=beat_times, x_coords=beat_times, ax=ax[1])
            ax[1].set(title='Recurrence similarity')
            ax[1].label_outer()
            librosa.display.specshow(X, y_axis='time', y_coords=beat_times, ax=ax[0])
            ax[0].set(title='Structure components')
            plt.show()

        seg_ids = self.clustering(X, beat_times, Rf, show)

        bound_beats = 1 + np.flatnonzero(seg_ids[:-1] != seg_ids[1:])
        # Count beat 0 as a boundary
        bound_beats = librosa.util.fix_frames(bound_beats, x_min=0)
        # Compute the segment label for each boundary
        bound_segs = list(seg_ids[bound_beats])
        # Convert beat indices to frames
        bound_frames = beats[bound_beats]
        # Make sure we cover to the end of the track
        bound_frames = librosa.util.fix_frames(bound_frames, x_min=None, x_max=C.shape[1]-1)

        bound_times = librosa.frames_to_time(bound_frames)
        freqs = librosa.cqt_frequencies(n_bins=C.shape[0], fmin=librosa.note_to_hz('C1'),
                                        bins_per_octave=self.BINS_PER_OCTAVE)
        fig, ax = plt.subplots()
        librosa.display.specshow(C, y_axis='cqt_hz', sr=self.sr, bins_per_octave=self.BINS_PER_OCTAVE,
                                 x_axis='time', ax=ax)
        intervals = list()
        for interval, label in zip(zip(bound_times, bound_times[1:]), bound_segs):
            ax.add_patch(patches.Rectangle((interval[0], freqs[0]), interval[1] - interval[0], freqs[-1],
                                           facecolor=self.colors(label), alpha=0.50))
            intervals.append(interval)
        if show:
            plt.show()
        return intervals, bound_segs
