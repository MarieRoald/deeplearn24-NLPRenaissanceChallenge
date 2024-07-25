import logging
from argparse import ArgumentParser
from pathlib import Path
from shutil import copy

import pandas as pd
from deeplearn24.utils import setup_logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_df(csv_path: Path) -> pd.DataFrame:
    """Load csv and filter out test split images if any"""
    logger.info(f"Reading {csv_path}")
    df = pd.read_csv(csv_path)
    if "split" in df.columns:
        df = df[df.split != "test"].copy()
        df.index = range(len(df))

    if not all(df.file_name.apply(Path).apply(lambda x: x.exists())):
        df["file_name"] = df.file_name.apply(lambda x: csv_path.parent / x)
        if not all(df.file_name.apply(lambda x: x.exists())):
            logger.error(f"Some source images do not exist")
            logger.info(df.file_name[0])
            exit()
    return df


def data_to_tesstrain(df: pd.DataFrame, tesstrain_gt_path: Path) -> pd.DataFrame:
    """Copy the image files in the DataFrame to the tesstrain data directory on the required format"""

    logger.info("Copying images to testrain directory")
    df["source_img_path"] = df.file_name.apply(Path)

    df["text_file"] = df.source_img_path.apply(lambda img: tesstrain_gt_path / f"{img.stem}.gt.txt")
    df["img_file"] = df.source_img_path.apply(lambda img: tesstrain_gt_path / f"{img.stem}.png")

    for e in df.itertuples():
        with e.text_file.open("w+") as f:
            f.write(str(e.transcription))
        copy(src=e.source_img_path, dst=e.img_file)

    return df


def create_list_files(df: pd.DataFrame, tesstrain_model_path: Path):
    """Copy custom split to tesstrain directory"""
    logger.info("Copying custom train and val split to tesstrain")

    tesstrain_path = tesstrain_model_path.parent.parent
    df["list_path"] = df.img_file.apply(
        lambda x: f"{x.relative_to(tesstrain_path)}"[:-4] + ".lstmf"
    )

    train = df[df.split == "train"]
    val = df[df.split == "val"]

    with open(tesstrain_model_path / "list.train", "w+") as f:
        f.write("\n".join(train.list_path))

    with open(tesstrain_model_path / "list.eval", "w+") as f:
        f.write("\n".join(val.list_path))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data_csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--tesstrain_directory",
        type=Path,
        default=Path(__file__).parent / "tesstrain/",
    )
    parser.add_argument("--model_name", required=True)
    parser.add_argument(
        "--copy_splits",
        action="store_true",
        help="Will create list.train and list.eval files in tesstrain model directory",
    )
    args = parser.parse_args()

    setup_logging()

    tesstrain_gt_path = args.tesstrain_directory / f"data/{args.model_name}-ground-truth"
    tesstrain_gt_path.mkdir(parents=True, exist_ok=True)

    df = get_df(args.data_csv)

    df = data_to_tesstrain(df=df, tesstrain_gt_path=tesstrain_gt_path)

    if args.copy_splits:
        if "split" not in df.columns:
            logger.error(f"--copy splits_flagged, but no split columns in csv-file")
            exit()
        tesstrain_model_path = args.tesstrain_directory / f"data/{args.model_name}"
        tesstrain_model_path.mkdir(parents=True, exist_ok=True)
        create_list_files(df=df, tesstrain_model_path=tesstrain_model_path)
