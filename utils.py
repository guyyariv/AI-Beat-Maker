import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from scipy.io import wavfile
import scipy.io


def convert_mp3_file_to_wav_file(src, dst):
    """
    :param file_name: mp3 file
    :param dst: destination wav file name
    """
    sound = AudioSegment.from_mp3(src)
    sound.export(dst, format="wav")


def audio_file_to_vector(file_name):
    """
    :param file_name: wav file
    :return: rate, audio data
    """
    return scipy.io.wavfile.read(file_name)
