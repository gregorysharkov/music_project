import re
from pathlib import Path
from typing import Dict

import music21 as mu
import pandas as pd
from tqdm import tqdm

from src.utils.corpus_extraction_utils import extract_notes

bwv295 = mu.corpus.parse('bach/bwv295')

path = r"partitions\What_is_autumn__DDT.mxl"


def simplify_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """function simplifies column_names"""

    new_col_names = [re.sub(r"[^a-z]", "_", x.lower()) for x in data.columns]
    data.columns = new_col_names  # type: ignore
    return data


def generate_data(path: Path, composer: str) -> pd.DataFrame:
    """function generates dataframe from path given"""
    file = mu.converter.parse(path)

    notes = extract_notes(file)  # type: ignore
    parsed_df = pd.DataFrame(notes)
    parsed_df["measure"] = range(len(parsed_df))
    parsed_df["composer"] = composer
    parsed_df["corpus"] = path.stem
    parsed_df = simplify_column_names(parsed_df)

    return parsed_df


def process_composer(composer_name: str) -> Dict:
    """function processes music21 composer corpus and returns a dictionary with all dataframes"""

    corpus_list = mu.corpus.getComposer('bach')
    corpus_list = [x for x in corpus_list if x.suffix == ".mxl"]
    corpus_dict = {f"bach_{x.stem}": generate_data(
        x, "bach") for x in tqdm(corpus_list)}
    return corpus_dict


if __name__ == "__main__":
    bach_dict = process_composer("bach")
    for key, value in bach_dict.items():
        value.to_csv(f"data\\00_raw\\bach\\{key}.csv", sep=";", index=0)
# file.show()
