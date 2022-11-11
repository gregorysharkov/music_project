import logging
from functools import reduce
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

import src.utils.training as tu

INPUT_PATH = Path("data\\01_preprocessed\\bach.csv")
COLUMNS = ["corpus", "measure", "node_id", "pitch", "duration"]
SEQ_LENGTH = 50
TRAINING_BATCH = 64


def main():
    logging.info("Loading data")
    data = tu.load_dataset(INPUT_PATH)
    tenors = tu.preprocess_dataset(data, COLUMNS, "tenor")
    logging.info("Transforming data into tensors")
    pitch_seq, duration_seq = tu.extract_pithces(tenors, SEQ_LENGTH)
    pitch_data = tu.generate_tensor_dataset(pitch_seq, tu.ids_from_pitches)
    duration_data = tu.generate_tensor_dataset(
        duration_seq, tu.ids_from_durations)
    print(f"{pitch_data=}")
    print(f"{duration_data=}")

    pitch_dataset = tu.map_split_function(pitch_data, TRAINING_BATCH)
    duration_dataset = tu.map_split_function(duration_data, TRAINING_BATCH)
    print(f"{pitch_dataset=}")
    print(f"{duration_dataset=}")


if __name__ == '__main__':
    main()
