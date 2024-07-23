import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from deeplearn24.utils import setup_logging
from jiwer import cer, wer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def evaluate(df: pd.DataFrame, output_dir: Path):
    df["transcription"] = df.transcription.apply(str)
    df["prediction"] = df.prediction.apply(str).apply(str.strip)  # move to post_processing?

    df["correct_prediction"] = df.transcription == df.prediction
    df["wer"] = wer(reference=df.transcription.to_list(), hypothesis=df.prediction.to_list())
    df["cer"] = cer(reference=df.transcription.to_list(), hypothesis=df.prediction.to_list())
    df.to_csv(output_dir / "line_results.csv", index=False)

    scores = {}

    scores["mean_wer"] = df.wer.mean()
    scores["mean_cer"] = df.cer.mean()
    scores["accuracy"] = len(df[df.correct_prediction]) / len(df)

    concat_predictions = " ".join(df.prediction)
    concat_transcriptions = " ".join(df.transcription)
    scores["concat_wer"] = wer(reference=concat_transcriptions, hypothesis=concat_predictions)
    scores["concat_cer"] = cer(reference=concat_transcriptions, hypothesis=concat_predictions)
    with open(output_dir / "overall_results.json", "w+") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--input_csv",
        type=Path,
        help=".csv-file containing transcriptions and predictions",
        required=True,
    )
    parser.add_argument("--output_dir", type=Path, help="directory to store results", required=True)
    parser.add_argument(
        "--remove_test",
        action="store_true",
        help="Will sort out csv rows with 'test' in image_path if flagged",
    )
    args = parser.parse_args()

    setup_logging()

    df = pd.read_csv(args.input_csv)
    if args.remove_test:
        df = df[df.image_path.apply(lambda x: "test" not in x)]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    evaluate(df, args.output_dir)
    logger.info(f"Done. See results at {args.output_dir}")
