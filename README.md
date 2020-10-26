## Requirements 

Python3:

> https://www.python.org/downloads/

patterns5.txt file containing input data. You should find one inside directory after unzipping. 

## Instalation

In root directory

> python3 -m venv venv

#### Windows:

> venv\Scripts\activate.bat

#### Linux/Osx:

> source venv/bin/activate

## Installing needed modules
> pip install -r requirements.txt



## RUN
In console:
> python neuron.py

## CONFIGURATION
to change parameters of the program look into .env file. Input formats:
- EPOCHS = {integer}
- LEARNING_RATE = {float}
- INPUT_WEIGHTS = {float,float}
- FILE = {string_file_path}
