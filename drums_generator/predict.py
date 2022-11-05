import pickle
from mido import Message, MidiFile, MidiTrack, MetaMessage, bpm2tempo
from keras.layers import BatchNormalization as BatchNorm
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
import numpy as np


import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))


def generate(tempo=120, length=2000):
    """ Generates the midi file """
    #load the notes used to train the model
    with open('data/notes', 'rb') as filepath:
        notes = pickle.load(filepath)

    # Get all pitch names
    pitchnames = sorted(set(item for item in notes))
    # Get all pitch names
    n_vocab = len(set(notes))

    network_input, normalized_input = prepare_sequences(notes, pitchnames, n_vocab)
    model = create_network(normalized_input, n_vocab)
    prediction_output = generate_notes(model, network_input, pitchnames,
                                       n_vocab, length)
    return create_midi(prediction_output, tempo)


def prepare_sequences(notes, pitchnames, n_vocab):

    """ Prepare the sequences used by the Neural Network """

    # map back from integers to notes
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    sequence_length = 30
    network_input = []
    output = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)
    # reshape the input into a format compatible with LSTM layers
    normalized_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    normalized_input = normalized_input / float(n_vocab)

    return (network_input, normalized_input)


def create_network(network_input, n_vocab):
    """ create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3,))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(264))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    model.load_weights('weights/weights-improvement-100-0.3411-bigger.hdf5')

    return model


def generate_notes(model, network_input, pitchnames, n_vocab, length=1200):

    """ Generate notes from the neural network based on a sequence of notes """

    # pick a random sequence from the input as a starting point for the prediction
    start = np.random.randint(0, len(network_input)-1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = network_input[start]
    prediction_output = []

    for note_index in range(length):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        index = np.argmax(prediction) # numpy array of predictions
        result = int_to_note[index] # indexing the note with the highest probability
        prediction_output.append(result) # that note is the prediction output

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output


def create_midi(prediction_output, tempo):
    mid = MidiFile()
    track = MidiTrack()
    track_meta = MidiTrack()
    mid.tracks.append(track_meta)
    mid.tracks.append(track)
    track_meta.append(MetaMessage('set_tempo', tempo=bpm2tempo(tempo)))
    track.append(Message('program_change', channel=9, program=12, time=0))

    for pattern in prediction_output:
        patterns = pattern.split(',')
        if patterns[0] == 'program_change':
            track.append(Message(patterns[0], channel=9, program=int(patterns[1]), time=int(patterns[2])))
        elif patterns[0] == 'control_change':
            track.append(Message(patterns[0], channel=9, value=int(patterns[1]), time=int(patterns[2])))
        else:
            track.append(Message(patterns[0], channel=9, note=int(patterns[1]), time=int(patterns[2]), velocity=int(patterns[3])))

    return mid


if __name__ == '__main__':
    generate()