from typing import List

import numpy as np
import tensorflow as tf


def generate_pitch_vocab() -> List[str]:
    """function generates pitch vocabulary"""
    octaves = range(1, 7, 1)
    accents = ["", "#", "##", "-", "--"]
    pitches = ["A", "B", "C", "D", "E", "F", "G"]

    pitch_vocab = []
    for pitch in pitches:
        for accent in accents:
            for octave in octaves:
                pitch_vocab.append(f"{pitch}{accent}{octave}")

    pitch_vocab.append("P")
    return pitch_vocab


def generate_duration_vocab() -> List[str]:
    """function generates duration vocabulary"""

    return [f"{x:.4f}" for x in np.arange(.0, 8.1, .125)]


pitch_vocab = generate_pitch_vocab()
duration_vocab = generate_duration_vocab()

ids_from_pitches = tf.keras.layers.StringLookup(
    vocabulary=pitch_vocab, mask_token=None
)

pitches_from_ids = tf.keras.layers.StringLookup(
    vocabulary=ids_from_pitches.get_vocabulary(), invert=True, mask_token=None,
)

ids_from_durations = tf.keras.layers.StringLookup(
    vocabulary=duration_vocab, mask_token=None
)

durations_from_ids = tf.keras.layers.StringLookup(
    vocabulary=ids_from_durations.get_vocabulary(), invert=True, mask_token=None
)
