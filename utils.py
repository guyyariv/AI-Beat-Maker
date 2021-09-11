import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from scipy.io import wavfile
import scipy.io


def create_wav_file_from_mp3_file(src, dst):
    """
    :param file_name: mp3 file
    :param dst: destination wav file name
    """
    sound = AudioSegment.from_mp3(src)
    sound.export(dst, format="wav")
