import logging
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import torch
from deeplearn24.utils import setup_logging
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from tqdm import trange

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def trocr_transcribe(
    model_path: Path, processor: str | Path, dataset_path: Path, split: str, output_csv: Path
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and processor
    model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)
    processor = TrOCRProcessor.from_pretrained(processor)

    # Set generation parameters
    model.generation_config.eos_token_id = processor.tokenizer.sep_token_id
    model.generation_config.max_length = 128
    model.generation_config.early_stopping = False
    model.generation_config.no_repeat_ngram_size = 3
    model.generation_config.length_penalty = 2.0
    model.generation_config.num_beams = 4

    # Load data for specified split
    df = pd.read_csv(dataset_path / "metadata.csv").rename(columns={"file_name": "image_path"})
    split_mask = df["image_path"].map(lambda s: s.startswith(split))
    df = df.loc[split_mask]

    images = [Image.open(dataset_path / image_path) for image_path in df["image_path"]]

    # Run inference
    logger.info("Preprocessing images")
    pixel_values = processor(images, return_tensors="pt")["pixel_values"].to(device)
    logger.info("Generating tokens, image size: pixel_values.shape")
    estimated_tokens = [model.generate(pixel_values[i].unsqueeze(0)) for i in trange(len(images))]
    logger.info("Decoding tokens")
    df["prediction"] = [
        processor.batch_decode(estimated_tokens[i], skip_special_tokens=True)[0]
        for i in trange(len(images))
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to .keras model used for inference",
    )
    parser.add_argument(
        "--processor",
        type=Path,
        required=True,
        help="Path to .keras model used for inference",
    )
    parser.add_argument(
        "--dataset", type=Path, help=".csv-file containing image info ", required=True
    )
    parser.add_argument("--split", type=str, help="train, test or val", default="val")
    parser.add_argument("--output_csv", type=Path, help="Output csv filename", required=True)
    args = parser.parse_args()

    setup_logging()

    if not args.model.exists():
        logger.error(f"{args.model} does not exist")
        exit()

    if not args.dataset.exists():
        logger.error(f"{args.input_csv} does not exist")
        exit()

    trocr_transcribe(
        model_path=args.model,
        processor=args.processor,
        dataset_path=args.dataset,
        split=args.split,
        output_csv=args.output_csv,
    )
    logger.info(f"Done. See predictions at {args.output_csv}")
