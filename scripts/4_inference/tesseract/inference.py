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


def tesseract_transcribe(model_name: str, df: pd.DataFrame, output_csv: Path):
    predictions = [
        pytesseract.image_to_string(Image.open(e.file_name), lang=model_name)
        for e in tqdm(df.itertuples(), total=len(df))
    ]
    df["prediction"] = predictions
    output_csv.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(output_csv, index=False)


def get_df(input_csv: Path, split: str = None) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    if split:
        df = df[df.split == split]

    if not all(df.file_name.apply(Path).apply(lambda x: x.exists())):
        df["file_name"] = df.file_name.apply(lambda x: input_csv.parent / x)
        if not all(df.file_name.apply(lambda x: x.exists())):
            logger.error(f"Some source images do not exist")
            exit()
    return df


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
    parser.add_argument(
        "--split",
        help="Data split to use (will filter out rows where column 'split' is not <split>)",
    )
    args = parser.parse_args()

    setup_logging()

    df = get_df(args.input_csv)

    tesseract_transcribe(model_name=args.model_name, df=df, output_csv=args.output_csv)
    logger.info(f"Done. See predictions at {args.output_csv}")
