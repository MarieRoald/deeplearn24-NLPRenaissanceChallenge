import logging
from argparse import ArgumentParser
from pathlib import Path
from shutil import copy

import pandas as pd
from deeplearn24.utils import setup_logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def data_to_tesstrain(csv_path: Path, tesstrain_gt_path: Path):
    """Copy the image files in the csv to the tesstrain data directory on the required format"""

    df = pd.read_csv(csv_path)
    logger.info(f"Reading {csv_path}")
    for e in df.itertuples():
        img = Path(e.image_path)
        if not img.exists():
            logger.error(f"{img} does not exist")
            exit()
        transcription = e.transcription
        txt_file = tesstrain_gt_path / f"{img.stem}.gt.txt"
        with txt_file.open("w+") as f:
            f.write(transcription)

        img_file = tesstrain_gt_path / f"{img.stem}.png"
        copy(src=img, dst=img_file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data_csv",
        type=Path,
        default=Path(__file__).parent.parent.parent.parent
        / "data/0_input/handout-modified/handout.csv",
    )
    parser.add_argument(
        "--tesstrain_directory",
        type=Path,
        default=Path(__file__).parent / "tesstrain/",
    )
    parser.add_argument("--model_name", required=True)
    args = parser.parse_args()

    setup_logging()

    tesstrain_gt_path = args.tesstrain_directory / f"data/{args.model_name}-ground-truth"
    tesstrain_gt_path.mkdir(parents=True, exist_ok=True)

    data_to_tesstrain(csv_path=args.data_csv, tesstrain_gt_path=tesstrain_gt_path)
