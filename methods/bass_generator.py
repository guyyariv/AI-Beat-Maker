import librosa
from mido import Message, MidiFile, MidiTrack, MetaMessage, bpm2tempo


def generate_bass(chords, tempo, times):
    """
    generate a bass line according to input chords, tempo and times and return a midi file
    """

    # notes = {'C:maj': 60, 'G:maj': 55, 'D:maj': 50, 'A:maj': 57,
    #           'E:maj': 52, 'B:maj': 59, 'F#:maj': 54, 'C#:maj': 61,
    #           'G#:maj': 56, 'D#:maj': 51, 'A#:maj': 58, 'F:maj': 53,
    #           'A:min': 57, 'E:min': 52, 'B:min': 59, 'F#:min': 54,
    #           'C#:min': 61, 'G#:min': 56, 'D#:min': 51, 'A#:min': 58,
    #           'F:min': 53, 'C:min': 60, 'G:min': 55, 'D:min': 50, 'N': 62}

    mid = MidiFile()
    track = MidiTrack()
    track_meta = MidiTrack()
    mid.tracks.append(track_meta)
    mid.tracks.append(track)
    track_meta.append(MetaMessage('track_name', name='Bass', time=0))
    track_meta.append(MetaMessage('set_tempo', tempo=bpm2tempo(tempo)))
    track.append(Message('program_change', channel=1, program=39, time=0))

    for i, chord in enumerate(chords):

        track.append(Message('note_on', channel=1, note=librosa.note_to_midi(chord.split(':')[0]) + 12, velocity=127, time=0))
        track.append(Message('note_off', channel=1, note=librosa.note_to_midi(chord.split(':')[0]) + 12, velocity=64, time=int(times[i])))

    return mid
