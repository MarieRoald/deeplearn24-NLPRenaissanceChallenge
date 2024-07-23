# Training tesseract models with tesstrain 

## Step1: install tesstrain and tesseract dev tools
Clone the [tesstrain repository](https://github.com/tesseract-ocr/tesstrain) and follow installation instructions in the README file. (If you've already installed tesseract you might be good to go after just cloning the repo)

## Step2: prepare data for training
Use `copy_data_to_tesstrain.py` to copy your training data to the tesstrain data directory

Ex :
```
pdm run python scripts/3_training/tesseract/copy_data_to_tesstrain.py --model_name spa_old_renai_15k --tesstrain_directory scripts/3_training/tesstrain --data_csv data/0_input/handout-modified/handout.csv
```

## Step3: set env variables
Run `tesseract --list-langs` to see where your tessdata directory is (e.g `/usr/local/share/tessdata/`)  
Set the `TESSDATA_PREFIX` environment variable to point to your tessdata directory:  
`export TESSDATA_PREFIX=[your-tessdata-directory]`

## Step4: train
Navigate the root of your tesstrain repo, train with make. (Below only the necessary basic arguments are listed, see all training arguments with `make help`)
Make sure that `[model_name]` corresponds to the `--model_name` argument used for `copy_data_to_tesstrain.py`

### To train from scratch:
```
make training MODEL_NAME=[model_name] TESSDATA=$TESSDATA_PREFIX
```

### To continue training existing tesseract model 
Run `tesseract --list-langs` to list available models.

You can download a language base model from the tesseract repo like this:  
`wget https://github.com/tesseract-ocr/tessdata_best/raw/main/[lang].traineddata -P $TESSDATA_PREFIX`  
Replace `[lang]` with the 3-charcter iso-langcode (and make sure a model for your language actually exists by browsing [the repo](https://github.com/tesseract-ocr/tessdata_best/)). 

Or move the .traineddata-file from one of your trained models to the tessdata repo to make it available to tesseract

Then:
```
make training MODEL_NAME=[model_name] START_MODEL=[lang] TESSDATA=$TESSDATA_PREFIX
```

