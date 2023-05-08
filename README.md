# Hand Gesture Recognition Project

**This project aims to put your understanding of machine learning algorithms into practice
with a real world problem.**
## Environment

you can download all the packages used using conda or pip
1. This will create new environment from scratch with all listed modules.

    conda env create -f environment.yml

2. If you already have an env and want to install the modules only you can use
The --prune option removes any packages that are not listed in the environment.yml file.

    conda env update --name <env_name> --file environment.yml --prune

3. Or using pip
    
    pip install -r requirements-pip.txt


## Dataset

### Option 1: download the processed data

download https://drive.google.com/drive/u/2/folders/1o9wzwaJVfrbpCFJ0rIyed1QvARh0JAtn

unzip the zipfile and put it all into `data folder`

### Option 2: run the data insertion script data_insertion.py

It will handle everything from downloading into extracting data into /data folder.

However you would need to have gdown library installed. (included in requirements.txt)

## Data Preparation
## Model Training
## Model Evaluation
