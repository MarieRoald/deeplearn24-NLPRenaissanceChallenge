import logging
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import pytesseract
from deeplearn24.utils import setup_logging
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# avoid debug logs from pillow
pil_logger = logging.getLogger("PIL")
pil_logger.setLevel(logging.INFO)


def tesseract_transcribe(model_name: str, input_csv: Path, output_csv: Path):
    df = pd.read_csv(input_csv)
    predictions = [
        pytesseract.image_to_string(Image.open(e.file_name), lang=model_name)
        for e in tqdm(df.itertuples(), total=len(df))
    ]
    df["prediction"] = predictions
    output_csv.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model_name",
        required=True,
        help="Name of tesseract model used for inference (see available models w 'tesseract --list-langs')",
    )
    parser.add_argument(
        "--input_csv", type=Path, help=".csv-file containing image info ", required=True
    )
    parser.add_argument("--output_csv", type=Path, help="Output csv filename", required=True)
    args = parser.parse_args()

    setup_logging()

    if not args.input_csv.exists():
        logger.error(f"{args.input_csv} does not exist")
        exit()

    tesseract_transcribe(
        model_name=args.model_name, input_csv=args.input_csv, output_csv=args.output_csv
    )
    logger.info(f"Done. See predictions at {args.output_csv}")
