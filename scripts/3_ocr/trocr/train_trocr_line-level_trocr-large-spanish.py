from functools import partial
from pathlib import Path

import datasets
import mlflow
import torch
from deeplearn24.mlflow_callbacks import ImageSaverCallback, TrainEvalCallback
from deeplearn24.postprocessing import (
    load_dictionaries,
    post_process_text,
)
from deeplearn24.tr_ocr import compute_metrics, transform_data
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    default_data_collator,
)

mlflow.set_tracking_uri("http://localhost:5400")
mlflow.set_experiment("TrOCR large-spanish Doc-UFCN")

train_set = datasets.load_dataset(
    "imagefolder",
    data_dir="/hdd/home/mariero/deeplearn24/data/2_bounding_box/Doc-UFCN_processed",
    split="train",
)

validation_set = datasets.load_dataset(
    "imagefolder",
    data_dir="/hdd/home/mariero/deeplearn24/data/2_bounding_box/Doc-UFCN_processed",
    split="validation",
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = TrOCRProcessor.from_pretrained("qantev/trocr-large-spanish")
model = VisionEncoderDecoderModel.from_pretrained("qantev/trocr-large-spanish").to(device)

tokens = processor.tokenizer(train_set["transcription"], padding="do_not_pad", max_length=128)
max_tokens = max(len(token) for token in tokens["input_ids"])
max_target_length = int(1.5 * max_tokens)
transform_data_partial = partial(
    transform_data, processor=processor, max_target_length=max_target_length
)
processed_train_set = train_set.with_transform(transform_data_partial)
processed_validation_set = validation_set.with_transform(transform_data_partial)

# set special tokens used for creating the decoder_input_ids from the labels
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
# make sure vocab size is set correctly
model.config.vocab_size = model.config.decoder.vocab_size

# set beam search parameters
model.generation_config.eos_token_id = processor.tokenizer.sep_token_id
model.generation_config.max_length = max_target_length
model.generation_config.early_stopping = False
model.generation_config.no_repeat_ngram_size = 3
model.generation_config.length_penalty = 2.0
model.generation_config.num_beams = 4

unique_words = load_dictionaries(
    Path("data/0_input/sbwce-corpus/dictionary.json"),
    Path("data/0_input/dataset_words/dictionary.json"),
)

with mlflow.start_run() as run:
    # Setup checkpoint dir
    experiment_name = Path(__file__).stem
    checkpoint_dir = Path(f"data/3_ocr/{experiment_name}/{run.info.run_name}/")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Setup trainer args
    eval_frequency = 20
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        eval_strategy="steps",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        fp16=False,
        output_dir=checkpoint_dir,
        logging_steps=2,
        save_steps=100,
        eval_steps=eval_frequency,
        remove_unused_columns=False,
        max_steps=1000,
        learning_rate=1e-5,
        metric_for_best_model="eval_cer",
        load_best_model_at_end=True,
        greater_is_better=False,
    )

    # Setup trainer
    postprocess = partial(post_process_text, unique_words=unique_words)
    eval_func = partial(compute_metrics, processor=processor, postprocess=postprocess)
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.feature_extractor,
        args=training_args,
        compute_metrics=eval_func,
        train_dataset=processed_train_set,
        eval_dataset=processed_validation_set,
        data_collator=default_data_collator,
        callbacks=[
            ImageSaverCallback(
                processor,
                validation_set,
                processed_validation_set,
                device,
                postprocess=postprocess,
                save_frequency=eval_frequency,
            ),
            TrainEvalCallback(eval_func, batch_size=50),
        ],
    )
    trainer.train()
    trainer.save_model(checkpoint_dir / "final_model")
    processor.save_pretrained(checkpoint_dir / "processor")


image = train_set[0]["image"]
pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
