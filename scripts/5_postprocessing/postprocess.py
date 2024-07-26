from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import pandas as pd
from deeplearn24.postprocessing import load_dictionaries, post_process_text

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--skip_rule_based_processing", action="store_true", default=False)
    parser.add_argument(
        "--dictionary_paths",
        type=Path,
        nargs="+",
        required=False,
        default=[
            Path("data/0_input/sbwce-corpus/dictionary.json"),
            Path("data/0_input/dataset_words/dictionary.json"),
        ],
    )

    args = parser.parse_args()
    dictionaries = load_dictionaries(*args.dictionary_paths)
    df = pd.read_csv(args.input_csv)
    df["prediction"] = df["prediction"].apply(
        partial(
            post_process_text,
            unique_words=dictionaries,
            include_rule_based_processing=not args.skip_rule_based_processing,
        )
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
