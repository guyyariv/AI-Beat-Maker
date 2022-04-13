import glob
import pickle
from keras.layers import BatchNormalization as BatchNorm
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from mido import MidiFile
import numpy as np


def train_network():

    """ This function calls all other functions and trains the LSTM"""

    notes = get_notes()

    # with open('data/notes', 'rb') as filepath:
    #     notes = pickle.load(filepath)

    # get amount of pitch names
    n_vocab = len(set(notes))

    network_input, network_output = prepare_sequences(notes, n_vocab)

    model = create_network(network_input, n_vocab)

    train(model, network_input, network_output)


def get_notes():
    notes = []

    for file in glob.glob("midi_drums/*.midi"):
        input_midi = MidiFile(file)

        for t in input_midi.tracks[-1]:
            # if 'track' in t.type:
            #     notes.append(t.type)
            if t.type == 'program_change':
                notes.append(f'{t.type},{t.program},{t.time}')
            elif t.type == 'control_change':
                notes.append(f'{t.type},{t.value},{t.time}')
            elif t.type == 'note_on' or t.type == 'note_off':
                notes.append(f'{t.type},{t.note},{t.time},{t.velocity}')
            else:
                continue
    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)
    return notes


def prepare_sequences(notes, n_vocab):

    """ Prepare the sequences which are the inputs for the LSTM """

    # sequence length should be changed after experimenting with different numbers
    sequence_length = 30

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

    # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)


def create_network(network_input, n_vocab):
    """ create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        recurrent_dropout=0.2,
        return_sequences=True
    ))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.2,))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.2))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.2))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    model.load_weights('weights/weights-improvement-03-3.8227-bigger.hdf5')

    return model


def train(model, network_input, network_output):

    """ train the neural network """

    filepath = "weights/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    # experiment with different epoch sizes and batch sizes
    model.fit(network_input, network_output, epochs=20, batch_size=256, callbacks=callbacks_list)


if __name__ == '__main__':
    train_network()
