import numpy as np
import librosa
from librosa.display import specshow
import matplotlib.pyplot as plt


def chord_rec(y, sr, plot=False):
    # Create templates for major, minor, and no-chord qualities
    maj_template = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    min_template = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])
    N_template = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.]) / 4.
    # Generate the weighting matrix that maps chroma to labels
    weights = np.zeros((25, 12), dtype=float)
    labels = ['C:maj', 'C#:maj', 'D:maj', 'D#:maj', 'E:maj', 'F:maj',
              'F#:maj', 'G:maj', 'G#:maj', 'A:maj', 'A#:maj', 'B:maj',
              'C:min', 'C#:min', 'D:min', 'D#:min', 'E:min', 'F:min',
              'F#:min', 'G:min', 'G#:min', 'A:min', 'A#:min', 'B:min', 'N']

    for c in range(12):
        weights[c, :] = np.roll(maj_template, c) # c:maj
        weights[c + 12, :] = np.roll(min_template, c)  # c:min
    weights[-1] = N_template  # the last row is the no-chord class
    # Make a self-loop transition matrix over 25 states
    trans = librosa.sequence.transition_loop(25, 0.5)

    # Load in audio and make features
    # Suppress percussive elements
    y = librosa.effects.harmonic(y, margin=4)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    # Map chroma (observations) to class (state) likelihoods
    probs = np.exp(weights.dot(chroma))  # P[class | chroma] ~= exp(template' chroma)
    probs /= probs.sum(axis=0, keepdims=True)  # probabilities must sum to 1 in each column
    # Compute independent frame-wise estimates
    chords_ind = np.argmax(probs, axis=0)
    # And viterbi estimates
    chords_vit = librosa.sequence.viterbi_discriminative(probs, trans)

    lst = np.array(labels)[chords_vit]
    res = [lst[0]]
    for i in range(1, len(lst)):
        if lst[i] == res[-1]:
            continue
        res.append(lst[i])

    if plot:
        # Plot the features and prediction map
        fig, ax = plt.subplots(nrows=2)
        specshow(chroma, x_axis='time', y_axis='chroma', ax=ax[0])
        specshow(weights, x_axis='chroma', ax=ax[1])
        ax[1].set(yticks=np.arange(25) + 0.5, yticklabels=labels, ylabel='Chord')

        # And plot the results
        fig, ax = plt.subplots()
        specshow(probs, x_axis='time', cmap='gray', ax=ax)
        times = librosa.times_like(chords_vit)
        ax.scatter(times, chords_ind + 0.25, color='lime', alpha=0.5, marker='+',
                   s=15, label='Independent')
        ax.scatter(times, chords_vit - 0.25, color='deeppink', alpha=0.5, marker='o',
                   s=15, label='Viterbi')
        ax.set(yticks=np.unique(chords_vit),
               yticklabels=[labels[i] for i in np.unique(chords_vit)])
        ax.legend()

        plt.show()

    return res


def rearrange_by_chord_recognition(sliced_audio, samp_rate):
    """
    This function will arrange the sliced sections randomly.
    """

    if not sliced_audio:
        raise Exception

    labels = ['C:maj', 'G:maj', 'D:maj', 'A:maj',
              'E:maj', 'B:maj', 'F#:maj', 'C#:maj',
              'G#:maj', 'D#:maj', 'A#:maj', 'F:maj',
              'A:min', 'E:min', 'B:min', 'F#:min',
              'C#:min', 'G#:min', 'D#:min', 'A#:min',
              'F:min', 'C:min', 'G:min', 'D:min', 'N']

    labels_dict = {l: i for i, l in enumerate(labels)}

    trans = np.zeros((25, 25))
    row = np.array([24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]) ** 3
    row = row / np.sum(row)
    for i in range(25):
        temp_row = np.roll(row, i) + np.abs(np.random.normal(0, 1e-4))
        trans[i, :] = temp_row / np.sum(temp_row)

    chords = list()
    for i, slice in enumerate(sliced_audio):
        c = chord_rec(slice, samp_rate)
        if not c:
            chords.append((None, None, i))
            continue
        chords.append((c[0], c[-1], i))

    start_ind = np.random.randint(0, len(chords))
    res = [start_ind]
    start_chord = chords[start_ind][1]
    del chords[start_ind]

    while chords:
        candidates = np.array(chords)[:, 0]
        probs = np.array([trans[labels_dict[start_chord], labels_dict[c]] for c in candidates])
        probs /= np.sum(probs)
        candidates = np.array(chords)[:, 2]
        res.append(int(np.random.choice(candidates, p=probs)))
        for i in range(len(chords)):
            if chords[i][2] == res[-1]:
                start_chord = chords[i][1]
                del chords[i]
                break

    rearranged = np.array([])

    for i in range(len(sliced_audio)):
        rearranged = np.append(rearranged, sliced_audio[res[i]])

    return rearranged
