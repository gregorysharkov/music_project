import numpy as np


def get_piano_notes():
    """function generates frequencies for all notes on a piano keyboard"""
    # White keys are in Uppercase and black keys (sharps) are in lowercase
    octave = ['C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g', 'A', 'a', 'B']
    base_freq = 440  # Frequency of Note A4
    keys = np.array([x+str(y) for y in range(0, 9) for x in octave])

    # Trim to standard 88 keys
    start = np.where(keys == 'A0')[0][0]
    end = np.where(keys == 'C8')[0][0]
    keys = keys[start:end+1]

    note_freqs = dict(
        zip(keys, [2**((n+1-49)/12)*base_freq for n in range(len(keys))]))
    note_freqs[''] = 0.0  # stop

    return note_freqs


def get_sine_wave(frequency, duration, sample_rate=44100, amplitude=4096):
    """ returns a dictionary that maps a note name to corresponding frequency in hertz"""
    t = np.linspace(0, duration, int(sample_rate*duration))  # Time axis
    wave = amplitude*np.sin(2*np.pi*frequency*t)
    return wave


def apply_overtones(frequency, duration, factor, sample_rate=44100, amplitude=4096):
    """function applies overtones"""

    assert abs(1-sum(factor)) < 1e-8

    frequencies = np.minimum(
        np.array([frequency*(x+1) for x in range(len(factor))]), sample_rate//2)
    amplitudes = np.array([amplitude*x for x in factor])

    fundamental = get_sine_wave(
        frequencies[0], duration, sample_rate, amplitudes[0])
    for i in range(1, len(factor)):
        overtone = get_sine_wave(
            frequencies[i], duration, sample_rate, amplitudes[i]
        )

        fundamental += overtone

    return fundamental


if __name__ == '__main__':
    from scipy.io import wavfile
    note_freqs = get_piano_notes()
    frequency = note_freqs.get("C4")

    sine_wave = get_sine_wave(frequency=frequency, duration=2, amplitude=2048)

    note = apply_overtones(frequency, duration=2.5, factor=factor)
    wavfile.write('pure_c.wav', rate=44100, data=sine_wave.astype(np.int16))
