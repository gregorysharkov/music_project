from pathlib import Path

import src.utils.corpus_preprocessing_utils as cp


def main():
    input_path = Path("data\\00_raw\\bach\\")
    all_files = cp.preprocess_data(input_path)
    all_files = cp.remove_bad_partitions(all_files)

    output_path = Path("data\\01_preprocessed\\bach.csv")
    all_files.to_csv(output_path, sep=";", index=False)


if __name__ == '__main__':
    main()
