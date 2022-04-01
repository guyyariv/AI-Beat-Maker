import glob
import pickle
import numpy
from keras.layers import BatchNormalization as BatchNorm
from music21 import converter, instrument, note, chord, stream
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from collections import Counter
import numpy as np

def train_network():

    """ This function calls all other functions and trains the LSTM"""

    notes = get_notes()

    # get amount of pitch names
    n_vocab = len(set(notes))

    network_input, network_output = prepare_sequences(notes, n_vocab)

    model = create_network(network_input, n_vocab)

    train(model, network_input, network_output)


# def get_notes():
#
#     """ Extracts all notes and chords from midi files in the ./hip_hop
#     directory and creates a file with all notes in string format"""
#
#     notes = []
#
#     for file in glob.glob("midi_songs/*.mid"):
#         midi = converter.parse(file)
#
#         print("Parsing %s" % file)
#
#         notes_to_parse = None
#
#         try: # file has instrument parts
#             s2 = instrument.partitionByInstrument(midi)
#             notes_to_parse = s2.parts[0].recurse()
#         except: # file has notes in a flat structure
#             notes_to_parse = midi.flat.notes
#
#         for element in notes_to_parse:
#             if isinstance(element, note.Note):
#                 notes.append(str(element.pitch))
#             elif isinstance(element, chord.Chord):
#                 notes.append('.'.join(str(n) for n in element.normalOrder))
#
#     with open('data/notes', 'wb') as filepath:
#         pickle.dump(notes, filepath)
#
#     return notes


def get_notes_chords_rests(instrument_type, path):
    try:
        midi = converter.parse(path)
        parts = instrument.partitionByInstrument(midi)
        note_list = []
        for music_instrument in range(len(parts)):
            print(path)
            print(parts.parts[music_instrument].id)
            if parts.parts[music_instrument].id in instrument_type:
                for element_by_offset in stream.iterator.OffsetIterator(parts[music_instrument]):
                    for entry in element_by_offset:
                        if isinstance(entry, note.Note):
                            note_list.append(str(entry.pitch))
                        elif isinstance(entry, chord.Chord):
                            note_list.append('.'.join(str(n) for n in entry.normalOrder))
                        # # elif isinstance(entry, note.Rest):
                        # #     note_list.append('Rest')
        return note_list
    except Exception as e:
        print("failed on ", path)
        pass


def get_notes():
    """ Get all the notes and chords from the midi files in the ./hip_hop directory """
    notes = []
    ex_set = set()
    for file in glob.glob("drums_generator/midi_drums/*.midi"):
        num = file.split('\\')[-1].split('_')[0]
        if num in ex_set:
            continue
        ex_set.add(num)
        print("Parsing %s" % file)
        notes += get_notes_chords_rests(['Percussion'], file)

    with open('drums_generator/data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes


def prepare_sequences(notes, n_vocab):

    """ Prepare the sequences which are the inputs for the LSTM """

    # sequence length should be changed after experimenting with different numbers
    sequence_length = 5

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
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
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
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3,))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    model.load_weights('drums_generator/weights/weights-improvement-117-0.4999-bigger.hdf5')

    return model

def train(model, network_input, network_output):

    """ train the neural network """

    filepath = "drums_generator/weights/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    # experiment with different epoch sizes and batch sizes
    model.fit(network_input, network_output, epochs=500, batch_size=64, callbacks=callbacks_list)

if __name__ == '__main__':
    train_network()
