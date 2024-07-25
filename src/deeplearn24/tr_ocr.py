from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypedDict

import evaluate

if TYPE_CHECKING:
    import torch
    import transformers.models.trocr.processing_trocr
    import transformers.trainer_utils


class InputData(TypedDict):
    image: torch.Tensor
    transcription: str


class TransformedData(TypedDict):
    pixel_values: torch.Tensor
    labels: list[int]


def transform_data(
    batch: InputData,
    processor: transformers.models.trocr.processing_trocr.TrOCRProcessor,
    max_target_length: int,
) -> TransformedData:
    processed_images = processor(images=batch["image"], return_tensors="pt").pixel_values

    labels = processor.tokenizer(
        batch["transcription"], padding="max_length", max_length=max_target_length
    ).input_ids

    # The torch.nn.modules.loss.CrossEntropyLoss has -100 as the default IgnoreIndex
    # So setting the PAD tokens to -100 will make sure they are ignored when we compute the loss
    labels = [label if label != processor.tokenizer.pad_token_id else -100 for label in labels]
    return {"pixel_values": processed_images, "labels": labels}


cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")


def compute_metrics(
    pred: transformers.trainer_utils.EvalPrediction,
    processor: transformers.models.trocr.processing_trocr.TrOCRProcessor,
    postprocess: Callable[[str], str],
) -> dict[str, float]:
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    if not isinstance(pred_str, list):
        pred_str_post = postprocess(pred_str)
    else:
        pred_str_post = [postprocess(text) for text in pred_str]

    out = {
        "cer": cer_metric.compute(predictions=pred_str, references=label_str),
        "wer": wer_metric.compute(predictions=pred_str, references=label_str),
        "cer_postprocessed": cer_metric.compute(predictions=pred_str_post, references=label_str),
        "wer_postprocessed": wer_metric.compute(predictions=pred_str_post, references=label_str),
    }

    lowest_cer = min(["cer", "cer_postprocessed"], key=out.__getitem__)
    lowest_wer = min(["wer", "wer_postprocessed"], key=out.__getitem__)
    out["cer_lowest"] = out[lowest_cer]
    out["wer_lowest"] = out[lowest_wer]
    out["cer_postprocessing_helped"] = lowest_cer == "cer_postprocessed"
    out["wer_postprocessing_helped"] = lowest_wer == "wer_postprocessed"

    return out
