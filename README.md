# deeplearn24

## Start MLFlow

```
sudo docker compose up mlflow -d
```

## Open MLFlow (via VPN)

[http://cicero.nb.no:5400](http://cicero.nb.no:5400)


## Log to MLFlow (via VPN)

```python
import mlflow

mlflow.set_tracking_uri("http://cicero.nb.no:5400")
mlflow.set_experiment("check-cicero-connection")

with mlflow.start_run():
    mlflow.log_metric("foo", 1)
    mlflow.log_metric("bar", 2)
```

## Source for book
https://bdh-rd.bne.es/viewer.vm?lang=en&id=0000194858&page=1


## Install PDFImages to extract images from the book PDF
```
sudo apt install poppler-utils
```

## Virtual environments
To run Doc-UFCN (line segmentation), we need really old versions of some packages (e.g. PyTorch), which hinders us from using many new libraries.
We have therefore created dependencies for multiple environments.

For example, to run line-level segmentation, we need to create a virtual environment with the Doc-UFCN dependencies:
```raw
pdm venv create --name doc_ufcn 3.11
pdm lock --group doc_ufcn && pdm install --venv doc_ufcn --with doc_ufcn
pdm run --venv doc_ufcn scripts/2_bounding_box/Doc-UFCN/segment_lines_docufcn.py
```

Similarly, to fit and evaluate TR-OCR models (and run postprocessing on them), we need a virtual environment with the TR-OCR dependencies
```raw
pdm venv create --name trocr 3.11
pdm lock --group trocr && pdm install --venv trocr --with trocr
pdm run --venv trocr scripts/3_ocr/trocr/train_trocr_line-level_trocr-large-spanish.py
pdm run --venv trocr python scripts/4_inference/trocr/inference.py \
    --model data/3_ocr/train_trocr_line-level_trocr-large-spanish/awesome-bass-898/final_model \
    --processor data/3_ocr/train_trocr_line-level_trocr-large-spanish/awesome-bass-898/processor \
    --dataset data/2_bounding_box/Doc-UFCN_processed \
    --split val \
    --output_csv data/4_inference/trocr/line-level_trocr-large-spanish.csv

# Evaluate without postprocessing
pdm run --venv trocr python scripts/6_evaluation/evaluate.py \
    --input_csv data/4_inference/trocr/line-level_trocr-large-spanish.csv \
    --output_dir data/6_evaluation/trocr/line-level_trocr-large-spanish

# Evaluate with postprocessing
pdm run --venv trocr python scripts/5_postprocessing/postprocess.py \
    --input_csv data/4_inference/trocr/line-level_trocr-large-spanish.csv
    --output_csv data/5_postprocessing/trocr/line-level_trocr-large-spanish.csv
pdm run --venv trocr python scripts/6_evaluation/evaluate.py \
    --input_csv data/5_postprocessing/trocr/line-level_trocr-large-spanish.csv
    --output_dir data/6_evaluation/trocr/line-level_trocr-large-spanish-postprocessed
```


## Useful PDM commands

Since Doc-UFCN requires old dependencies, and we don't want to use really old versions of PyTorch and Huggingface, we need multiple virtualenvs.
Some useful commands are:

 * Create new virtualenvironment: `pdm venv create --name {venv_name} {python-version}`
 * List all virtual environments: `pdm venv list`
 * Install dependency group in named venv: `pdm install --venv {venv_name} --with {group-name}`
 * Add dependencies to group: `pdm add --group {group_name} {dependencies}`
 * Update lock file with a specific dependency group (needed if dependency group is changed by manually updating pyproject.toml): `pdm lock --group {group-name}`
 * Update installed dependencies in a venv after manually updating pyproject.toml: `pdm sync --venv {venv_name}`

### Example: do things in doc_ufcn environment

 * Create environment: `pdm venv create --name doc_ufcn 3.11`
 * Install dependencies: `pdm install --venv doc_ufcn --with doc_ufcn`
 * Run command: `pdm run --venv doc_ufcn python`
 * Add dependencies `pdm add --group doc_ufcn matplotlib`

### Example: do things in trocr envifonment

 * Create environment: `pdm venv create --name trocr 3.11`
 * Install dependencies: `pdm install --venv trocr --with trocr`
 * Run command: `pdm run --venv trocr python`
 * Add dependencies `pdm add --group trocr datasets`

## VSCode tips:

If you go through all virtual envrionments you've created with PDM and set them as the selected interpreter in VSCode, then you can easily switch between them by pressing the virtual envrionment name in the bottom right corner.
To see the path off all PDM virtual environments for this project, type `pdm venv list`.

## License
The code (but not data) in this repository is licensed following a GPL3 license.
