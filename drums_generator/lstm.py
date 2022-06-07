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

    # with open('data/notes_trap', 'rb') as filepath:
    #     notes = pickle.load(filepath)

    # get amount of pitch names
    n_vocab = len(set(notes))

    network_input, network_output = prepare_sequences(notes, n_vocab)

    model = create_network(network_input, n_vocab)

    train(model, network_input, network_output)


def get_notes():
    notes = []

    for file in glob.glob("midi_drums/*"):
        input_midi = MidiFile(file)

        for t in input_midi.tracks:
            for m in t:
                if (not m.is_meta and m.type != 'sysex' and m.channel == 9) or (not m.is_meta and m.type != 'sysex' and file.endswith('mid')):
                    # if 'track' in t.type:
                    #     notes.append(t.type)
                    if m.type == 'program_change':
                        notes.append(f'{m.type},{m.program},{m.time}')
                    elif m.type == 'control_change':
                        notes.append(f'{m.type},{m.value},{m.time}')
                    elif m.type == 'note_on' or m.type == 'note_off':
                        notes.append(f'{m.type},{m.note},{m.time},{m.velocity}')
                    else:
                        continue
    with open('data/full_notes', 'wb') as filepath:
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
        1024,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(LSTM(1024, return_sequences=True, recurrent_dropout=0.3,))
    model.add(LSTM(1024))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    model.load_weights('weights/weights-improvement-60-0.3887-bigger.hdf5')

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
    model.fit(network_input, network_output, epochs=40, batch_size=512, callbacks=callbacks_list)


if __name__ == '__main__':
    train_network()
