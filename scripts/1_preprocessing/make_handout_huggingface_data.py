import logging
from pathlib import Path
from shutil import copy

import pandas as pd
from deeplearn24.utils import setup_logging
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def rename_handout_data():
    logger.info(
        "Copying data from data/0_input/handout-created to data/0_input/handout-huggingface/"
    )

    df = pd.read_csv("data/0_input/handout-created/outputsTest.csv")

    df_test = df[df.IDENTITY.apply(str.isnumeric)].copy()
    df_test["split"] = ["test"] * len(df_test)

    df_train = pd.read_csv("data/0_input/handout-created/train.csv")
    df_train["split"] = ["train"] * len(df_train)

    df_val = pd.read_csv("data/0_input/handout-created/valid.csv")
    df_val["split"] = ["val"] * len(df_val)

    df = pd.concat([df_test, df_train, df_val])
    df["from_path"] = df.FILENAME.apply(lambda x: f"data/0_input/handout-created/outputTest/{x}")

    df["file_name"] = df.apply(
        lambda row: f"data/0_input/handout-huggingface/{row.split}/{row.name}.png", axis=1
    )

    df["transcription"] = [e.IDENTITY if e.split != "test" else "" for e in df.itertuples()]

    modified_p = Path("data/0_input/handout-huggingface/")
    test_p = modified_p / "test"
    train_p = modified_p / "train"
    val_p = modified_p / "val"

    test_p.mkdir(parents=True, exist_ok=True)
    train_p.mkdir(parents=True, exist_ok=True)
    val_p.mkdir(parents=True, exist_ok=True)

    for e in tqdm(df.itertuples()):
        img = Path(e.from_path)
        if not img.exists():
            logger.error(f"{img} does not exist")
            exit()
        copy(src=img, dst=e.file_name)

    df["file_name"] = df.file_name.apply(
        lambda x: x.replace("data/0_input/handout-huggingface/", "")
    )
    df[["file_name", "transcription"]].to_csv(
        "data/0_input/handout-huggingface/metadata.csv", index=False
    )


if __name__ == "__main__":
    setup_logging()

    rename_handout_data()

    logger.info("Done. See results in data/0_input/handout-huggingface")
