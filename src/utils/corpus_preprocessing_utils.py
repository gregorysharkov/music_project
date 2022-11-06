from ast import literal_eval
from functools import reduce
from operator import itemgetter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

BAD_CORPUS_LIST = ["bwv248.9-1", "bwv846", "bwv248.23-2"]


def transform_data(data: pd.DataFrame) -> pd.DataFrame:
    """function performs all transformations so that all partitures have the same structure"""

    static_cols = ["measure", "composer", "corpus"]

    # transform strings into arrays
    partition_cols = [x for x in data.columns if x not in static_cols]
    for col in partition_cols:
        data[col] = data[col].apply(lambda x: literal_eval(x))

    # melt arrays
    data = data.\
        melt(
            id_vars=static_cols, var_name="partition", value_name="notes"
        ).\
        explode("notes")

    na_mask = data.notes.isnull()
    data.loc[na_mask, "notes"] = data.loc[na_mask,
                                          "notes"].fillna("[None, None]").apply(eval)

    # extract notes and duration
    data["node_id"] = range(len(data))
    split = data["notes"].transform(
        {"pitch": itemgetter(0), "duration": itemgetter(1)})
    data["pitch"] = split.pitch
    data["duration"] = split.duration.apply(lambda x: f"{x:.4f}")

    return data


def preprocess_data(path: Path) -> pd.DataFrame:
    """function preprocesses files from a given directory"""

    all_files = []
    for path in tqdm(path.iterdir()):
        if path.is_file():
            data = pd.read_csv(path, sep=";")
            all_files.append(transform_data(data))

    return reduce(lambda x, y: pd.concat([x, y]), all_files)


def remove_bad_partitions(all_files: pd.DataFrame) -> pd.DataFrame:
    """function removes non-standard partitions from the given dataframe"""

    all_files = all_files[
        all_files.corpus.apply(lambda x: x not in BAD_CORPUS_LIST)
    ]

    all_files.pitch = all_files.pitch.fillna("P")

    return all_files
