from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from operator import attrgetter
from typing import TYPE_CHECKING, TypedDict

import evaluate
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import transformers
from tqdm import tqdm
from transformers.trainer_callback import TrainerCallback

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Generator

    import accelerate
    import datasets
    import PIL.Image
    from typing_extensions import Unpack  # Python 3.12 can use Unpack for TypedDicts

cer_metric = evaluate.load("cer", trust_remote_code=True)


@dataclass
class EvaluatedExample:
    cer: float
    image: PIL.Image.Image
    transcription: str
    estimated_transcription: str

    def visualise(self, ax: plt.Axes) -> None:
        ax.imshow(self.image)
        ax.set_title(
            f"Transcription: {self.transcription}\n"
            f"Estimated: {self.estimated_transcription}\n"
            f"CER: {self.cer}"
        )
        ax.axis("off")


class CallbackKwargs(TypedDict):
    tokenizer: transformers.models.vit.image_processing_vit.ViTImageProcessor
    optimizer: accelerate.optimizer.AcceleratedOptimizer
    lr_scheduler: torch.optim.lr_scheduler.LambdaLR
    train_dataloader: accelerate.data_loader.DataLoaderShard
    eval_dataloader: accelerate.data_loader.DataLoaderShard | None


class TrainEvalCallback(TrainerCallback):
    def __init__(
        self,
        compute_metrics: Callable[[transformers.trainer_utils.EvalPrediction], dict[str, float]],
        batch_size: int,
    ) -> None:
        self.compute_metrics = compute_metrics
        self.batch_size = batch_size
        self.rng = np.random.default_rng(42)

    def on_step_end(
        self,
        args: transformers.training_args_seq2seq.Seq2SeqTrainingArguments,
        state: transformers.trainer_callback.TrainerState,
        control: transformers.trainer_callback.TrainerControl,
        model: transformers.modeling_utils.PreTrainedModel,
        **kwargs: Unpack[CallbackKwargs],
    ) -> None:
        if not control.should_evaluate:
            return
        # Sample randomly from the dataset
        dataset = kwargs["train_dataloader"].dataset
        indices = self.rng.choice(len(dataset), self.batch_size, replace=False).tolist()

        metrics = {}
        sample = dataset[indices]

        # Unpack sampled dataset and move to device before running inference
        pixel_values = sample["pixel_values"].to(model.device)
        labels = sample["labels"]
        prediction = model.generate(pixel_values).tolist()

        # Compute metrics
        train_eval_prediction = transformers.trainer_utils.EvalPrediction(
            predictions=prediction, label_ids=labels.copy(), inputs=pixel_values
        )

        metrics_mean = {
            f"train_{k}": v for k, v in self.compute_metrics(train_eval_prediction).items()
        }

        mlflow.log_metrics(metrics_mean, step=state.global_step)


class ImageSaverCallback(TrainerCallback):
    def __init__(
        self,
        processor: transformers.processing_utils.ProcessorMixin,
        validation_data: datasets.arrow_dataset.Dataset,
        processed_validation_data: datasets.arrow_dataset.Dataset,
        device: torch.device,
        save_frequency: int = 10,
    ):
        self.processor = processor
        self.validation_data = validation_data
        self.processed_validation_data = processed_validation_data
        self.device = device
        self.save_frequency = save_frequency

        rng = np.random.default_rng(42)
        self.indices = rng.choice(len(self.validation_data), 5, replace=False)

    @cached_property
    def val_images(self) -> list[PIL.Image.Image]:
        return [val_data["image"] for val_data in self.validation_data]

    @cached_property
    def val_transcriptions(self) -> list[str]:
        return [val_data["transcription"] for val_data in self.validation_data]

    @cached_property
    def val_pixel_values(self) -> list[torch.Tensor]:
        return torch.concat(
            [
                processed_val_data["pixel_values"].unsqueeze(0).to(self.device)
                for processed_val_data in self.processed_validation_data
            ],
            axis=0,
        )

    def get_estimated_val_text(
        self, model: transformers.modeling_utils.PreTrainedModel
    ) -> list[str]:
        all_estimated_ids = model.generate(self.val_pixel_values)
        return [
            self.processor.batch_decode(generated_id, skip_special_tokens=True)[0]
            for generated_id in tqdm(
                all_estimated_ids,
                desc="Decoding on all validation examples",
                total=len(self.val_images),
            )
        ]

    def get_evaluated_examples(
        self, model: transformers.modeling_utils.PreTrainedModel
    ) -> Generator[EvaluatedExample, None, None]:
        all_estimated_text = self.get_estimated_val_text(model)
        for i in range(len(self.validation_data)):
            image = self.validation_data[i]["image"]
            estimated_text = all_estimated_text[i]
            transcription = self.validation_data[i]["transcription"]

            cer = cer_metric.compute(predictions=[estimated_text], references=[transcription])
            yield EvaluatedExample(cer, image, transcription, estimated_text)

    def on_step_end(
        self,
        args: transformers.training_args_seq2seq.Seq2SeqTrainingArguments,
        state: transformers.trainer_callback.TrainerState,
        control: transformers.trainer_callback.TrainerControl,
        model: transformers.modeling_utils.PreTrainedModel,
        **kwargs: Unpack[CallbackKwargs],
    ) -> None:
        if state.global_step % self.save_frequency != 0:
            return
        evaluated_examples = list(self.get_evaluated_examples(model))

        # Create a with the five constant examples
        fig, axs = plt.subplots(5, 1, figsize=(10, 11), tight_layout=True)

        for ax, idx in zip(axs, self.indices):
            evaluated_examples[idx].visualise(ax)

        mlflow.log_figure(fig, f"predictions_step_{state.global_step:04d}.png")
        plt.close(fig)

        # Create a figure with the five worst CER examples
        fig, axs = plt.subplots(5, 1, figsize=(10, 11), tight_layout=True)
        evaluated_examples = sorted(evaluated_examples, reverse=True, key=attrgetter("cer"))

        for evaluated_example, ax in zip(evaluated_examples, axs, strict=False):
            evaluated_example.visualise(ax)

        mlflow.log_figure(fig, f"worst_cer_step_{state.global_step:04d}.png")
        plt.close(fig)
