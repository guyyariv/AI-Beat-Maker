import ctypes.util
from midi2audio import FluidSynth

orig_ctypes_util_find_library = ctypes.util.find_library


def proxy_find_library(lib):
    if lib == 'fluidsynth':
        return 'libfluidsynth.so.1'
    else:
        return orig_ctypes_util_find_library(lib)

ctypes.util.find_library = proxy_find_library


def convert_midi_to_wav(midi_path, output_path, sound_font=None):
    fs = FluidSynth(sound_font='808.sf2') if sound_font else FluidSynth()
    fs.midi_to_audio(midi_path, output_path)
