import logging
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import pytesseract
import tensorflow as tf
from deeplearn24.utils import setup_logging
from example_model import predict_df
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def keras_transcribe(model: Path, input_csv: Path, output_csv: Path):
    if model.stem == "example_model":
        df = pd.read_csv(input_csv)
        df = predict_df(df)
        output_csv.parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(output_csv, index=False)
    else:
        logger.error("oops, not implemented")
        exit()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to .keras model used for inference",
    )
    parser.add_argument(
        "--input_csv", type=Path, help=".csv-file containing image info ", required=True
    )
    parser.add_argument("--output_csv", type=Path, help="Output csv filename", required=True)
    args = parser.parse_args()

    setup_logging()

    if not args.model.exists():
        logger.error(f"{args.model} does not exist")
        exit()

    if not args.input_csv.exists():
        logger.error(f"{args.input_csv} does not exist")
        exit()

    keras_transcribe(model=args.model, input_csv=args.input_csv, output_csv=args.output_csv)
    logger.info(f"Done. See predictions at {args.output_csv}")
