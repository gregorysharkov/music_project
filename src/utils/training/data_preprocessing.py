from functools import reduce
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
import tensorflow as tf
from tqdm import tqdm


def load_dataset(path: Path) -> pd.DataFrame:
    """function loads dataset from the given location"""

    return pd.read_csv(path, sep=";", dtype={"duration": str})


def preprocess_dataset(
    data: pd.DataFrame,
    columns: List[str],
    partition_name: str,
) -> pd.DataFrame:
    """
    function preprocesses dataset, extracts given 
    partition and makes it availbale for further use
    """

    return data[data.partition == partition_name][columns].\
        sort_values(["corpus", "measure", "node_id"])


def extract_pithces(data: pd.DataFrame, seq_length: int) -> Tuple[List[List[str]], List[List[str]]]:
    """function slices data into equal length sequences"""

    return_pitches = []
    return_durations = []
    for corpus in tqdm(data["corpus"].unique()):
        sliced_data = data[data.corpus == corpus]
        sliced_pitches = generate_slices(sliced_data.pitch, seq_length)
        sliced_durations = generate_slices(sliced_data.duration, seq_length)

        for pitch_seq, dur_seq in zip(sliced_pitches, sliced_durations):
            return_pitches.append(pitch_seq)
            return_durations.append(dur_seq)

    return return_pitches, return_durations


def generate_slices(sequence: pd.Series, seq_length: int) -> List[List[str]]:
    """function generates slices with step 1 from a given series of a given length"""

    if seq_length > len(sequence):
        return []

    return_list = []
    for i in range(len(sequence)-seq_length+1):
        return_list.append(list(sequence[i:i+seq_length]))

    return return_list


def generate_tensor_dataset(
    sequence: List[List[str]],
    id_converter: tf.keras.layers.StringLookup
) -> tf.data.Dataset:
    """function generates a tensor dataset from sequence slices"""
    combined_id_datasets = []
    for el in tqdm(sequence):
        ids = id_converter(el)
        ids_dataset = tf.data.Dataset.from_tensor_slices(ids).\
            batch(len(el), drop_remainder=True)
        combined_id_datasets.append(ids_dataset)

    return reduce(lambda x, y: x.concatenate(y), combined_id_datasets)


def split_input_target(sequence: Iterable) -> Tuple[Iterable]:
    """function splits sequence into input and target which is one element shifted"""

    input_seq = sequence[:-1]
    target_seq = sequence[1:]

    return input_seq, target_seq


def map_split_function(data: tf.data.Dataset, train_batch_size: int) -> tf.data.Dataset:
    """function splits all sequences into input and target"""

    return data.\
        map(split_input_target).\
        batch(train_batch_size, drop_remainder=True)


if __name__ == '__main__':
    sequence = pd.Series(["A", "B", "C", "D"])
    print(generate_slices(sequence, 2))
    print(generate_slices(sequence, 3))
    print(generate_slices(sequence, 4))
    print(generate_slices(sequence, 5))
