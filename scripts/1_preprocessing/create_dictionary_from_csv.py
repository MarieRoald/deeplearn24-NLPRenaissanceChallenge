import json
import unicodedata
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def create_dictionary(dataset_path: Path) -> frozenset[str]:
    """Creates a dictionary from the the train data in a csv file.
    Assumes that the csv file has a column named 'transcription' that contains the text data
    and a column named 'split' containing which split each transcription belongs to.
    """
    dictionary = set()

    df = pd.read_csv(dataset_path)
    for row in df.query("split == 'train'").itertuples():
        words = unicodedata.normalize("NFKD", row.transcription.strip().lower()).split()
        dictionary.update(set(words))

    return dictionary


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("output_path", type=Path)

    args = parser.parse_args()

    corpus = create_dictionary(args.csv_path)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as file:
        json.dump(list(corpus), file)
