# Inference with Tesseract 

## Step1: install Tesseract 5
Follow install instructions in [the official documentation](https://tesseract-ocr.github.io/tessdoc/Installation.html)  
Confirm installation with `tesseract --version`

## Step2: set TESSDATA_PREFIX env variable
Your tessdata directory may be something like `/usr/local/share/tessdata/` or `/usr/share/tesseract-ocr/5/tessdata/` depending on your installation.  
If you run `tesseract --list-langs` the path should follow after "List of available languages in"

Set the `TESSDATA_PREFIX` environment variable to point to your tessdata directory:  
`export TESSDATA_PREFIX=[your-tessdata-directory]`

## Step3: prepare model:
Run `tesseract --list-langs` to see the available models  
If you dont see the model you want to use, copy or move the `.traineddata` file of the model the tessdata directory.  

Example:  
`cp models/tesseract/spa_renai_10k.traineddata $TESSDATA_PREFIX`  
You may have to put sudo before cp to write to your tessdata repository

Or download a language base model from the tesseract repo like this:  
`wget https://github.com/tesseract-ocr/tessdata_best/raw/main/[lang].traineddata -P $TESSDATA_PREFIX`  
(Replace `[lang]` with the 3-charcter iso-langcode (and make sure a model for your language actually exists by browsing [the repo](https://github.com/tesseract-ocr/tessdata_best/)). Depending on access rights to the tessdata directory you might have to add `sudo` before `wget`. )
