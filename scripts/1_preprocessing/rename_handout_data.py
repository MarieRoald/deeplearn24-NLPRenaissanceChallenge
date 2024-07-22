import logging
from pathlib import Path
from shutil import copy

import pandas as pd
from deeplearn24.utils import setup_logging
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def rename_handout_data():
    logger.info("Copying data from data/0_input/handout-created to data/0_input/handout-modified/")

    df = pd.read_csv("data/0_input/handout-created/outputsTest.csv")
    df["is_test"] = df.IDENTITY.apply(str.isnumeric)
    df["from_path"] = df.FILENAME.apply(lambda x: f"data/0_input/handout-created/outputTest/{x}")
    df["image_path"] = [
        f"data/0_input/handout-modified/{'test' if e.is_test else 'train'}/{e.Index}.png"
        for e in df.itertuples()
    ]
    df["transcription"] = ["" if e.is_test else e.IDENTITY for e in df.itertuples()]

    modified_p = Path("data/0_input/handout-modified/")
    test_p = modified_p / "test"
    train_p = modified_p / "train"
    test_p.mkdir(parents=True, exist_ok=True)
    train_p.mkdir(parents=True, exist_ok=True)

    for e in tqdm(df.itertuples()):
        img = Path(e.from_path)
        if not img.exists():
            logger.error(f"{img} does not exist")
            exit()
        copy(src=img, dst=e.image_path)
        if not e.is_test:
            transcription = e.transcription

            txt_file = train_p / f"{Path(e.image_path).stem}.txt"
            with txt_file.open("w+") as f:
                f.write(transcription)

    df[["image_path", "transcription"]].to_csv(
        "data/0_input/handout-modified/handout.csv", index=False
    )
    df[~df.is_test][["image_path", "transcription"]].to_csv(
        "data/0_input/handout-modified/train.csv", index=False
    )
    df[df.is_test][["image_path"]].to_csv("data/0_input/handout-modified/test.csv", index=False)


if __name__ == "__main__":
    setup_logging()

    rename_handout_data()

    logger.info("Done")
