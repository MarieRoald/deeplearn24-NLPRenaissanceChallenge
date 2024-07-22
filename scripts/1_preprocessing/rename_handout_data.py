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
    modified_p = Path("data/0_input/handout-modified/words")
    modified_p.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv("data/0_input/handout-created/outputs.csv")
    df["from_path"] = df.FILENAME.apply(lambda x: f"data/0_input/handout-created/output/{x}")
    df["image_path"] = [f"data/0_input/handout-modified/words/{i}.png" for i in df.index]
    df["transcription"] = df.IDENTITY

    for e in tqdm(df.itertuples()):
        img = Path(e.from_path)
        if not img.exists():
            logger.error(f"{img} does not exist")
            exit()
        copy(src=img, dst=e.image_path)

        transcription = e.transcription

        txt_file = modified_p / f"{Path(e.image_path).stem}.gt.txt"
        with txt_file.open("w+") as f:
            f.write(transcription)

    df = df[["image_path", "transcription"]]
    df.to_csv("data/0_input/handout-modified/handout.csv", index=False)


if __name__ == "__main__":
    setup_logging()

    rename_handout_data()

    logger.info("Done")
