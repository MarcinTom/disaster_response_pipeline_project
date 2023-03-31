# Disaster Response Pipeline Project

## Table of Contents
1. [Project description](#description)
2. [Getting Started](#getting_started)
	1. [Dependencies](#dependencies)
	2. [Executing Program](#execution)
	3. [Project structure](#structure)

<a name="descripton"></a>
## Project description

This Project is part of Udacity Data Science Nanodegree Program. The dataset contains pre-labelled messages from real-life disaster events. The project aims to build a Natural Language Processing (NLP) model to categorize messages in a real time basis. Current solution uses Random Forest classifier, plus tuning done with GridSearch

Project parts:

1. Processing data, building an ETL pipeline to extract data from source, clean the data and save them in a SQLite DB
2. Build a machine learning pipeline to train a model which can classify text message in various categories
3. Run a web app which can show model results in real time

<a name="getting_started"></a>
## Getting Started

<a name="dependencies"></a>
### Dependencies
* Python 3.9+
* Machine learning: NumPy, SciPy, Pandas, Sciki-Learn
* NLP: NLTK
* SQLlite Database: SQLalchemy
* Model upload and download: Pickle
* Web app and graphs: Flask, Plotly

<a name="execution"></a>
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

<a name="structure"></a>
## File Description
~~~~~~~
        disaster_response_pipeline
          |-- app
                |-- templates
                        |-- go.html
                        |-- master.html
                |-- run.py
          |-- data
                |-- disaster_message.csv
                |-- disaster_categories.csv
                |-- DisasterResponse.db
                |-- process_data.py
                |-- ETL Pipeline Preparation.ipynb
          |-- models
                |-- classifier.pkl
                |-- train_classifier.py
                |-- ML Pipeline Preparation.ipynb
          |-- README
~~~~~~~
