import logging
from pathlib import Path
from shutil import copytree, rmtree

import pandas as pd
from deeplearn24.utils import setup_logging
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def compile_huggingface_dataset():
    logger.info(
        "Copying data from data/0_input/handout-created to data/0_input/handout-huggingface/"
    )

    data_dir = Path("data/0_input/handout-modified/our_splits")
    output_dir = Path("data/0_input/handout-huggingface-our_splits")
    rmtree(output_dir, ignore_errors=True)

    df_train = pd.read_csv(data_dir / "train.csv").assign(split="train")
    df_val = pd.read_csv(data_dir / "val.csv").assign(split="val")
    df_test = pd.read_csv(data_dir / "test.csv").assign(split="test", transcription="")
    df = pd.concat([df_test, df_train, df_val])

    df = pd.concat([df_test, df_train, df_val])
    df["file_name"] = df["image_path"].map(lambda s: Path(s).relative_to(data_dir))
    df = df.drop("image_path", axis=1)

    for split in ("train", "test", "val"):
        copytree(data_dir / split, output_dir / split)

    df.set_index("file_name").to_csv(output_dir / "metadata.csv")

    logger.info("Done. See results in %s", output_dir)


if __name__ == "__main__":
    setup_logging()

    compile_huggingface_dataset()
