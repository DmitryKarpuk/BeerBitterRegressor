# Beer bitterness prediction

## Description

This is a midterm project for ML Zoomcamp based on data from Kaggle competition ["How Bitter is the Beer"](https://www.kaggle.com/competitions/beer2020/overview) 

The International Bittering Units, or IBU, scale is used to quantify the bitterness of a beer. In this project, I've built a machine learning model and make a service to predict the IBU of a beer, given other attributes about the beer, including its color, alcohol content, and other features.

Exploratory Data Analysis of different items of beer have been done and published [there](https://github.com/DmitryKarpuk/BeerBitterRegressor/blob/master/src/beerbitterregressor/notebooks/EDA.ipynb).

For this issue were trained and estimated 4 different types of models:
- Ridge (linear least squares with l2 regularization.)
- RandomForestRegressor (ensemble of classifying decision trees fitted on various sub-samples)
- ExtraTreesRegressor (ensemble of randomized decision trees (a.k.a. extra-trees) fitted on various sub-samples)
- CatBoostRegressor  (algorithm for gradient boosting on decision trees)

In order to get best prediction I've used 3 ways of categorical features encoding:
- OneHotEncoding for Ridge, RandomForestRegressor and ExtraTreesRegressor ([notebook](https://github.com/DmitryKarpuk/BeerBitterRegressor/blob/master/src/beerbitterregressor/notebooks/OneHotEncoding.ipynb))
- TargetEncoding for Ridge, RandomForestRegressor and ExtraTreesRegressor ([notebook](https://github.com/DmitryKarpuk/BeerBitterRegressor/blob/master/src/beerbitterregressor/notebooks/TargetEncoding.ipynb))
- Ordered TargetEncoding for CatBoostRegressor ([notebook](https://github.com/DmitryKarpuk/BeerBitterRegressor/blob/master/src/beerbitterregressor/notebooks/Catboost.ipynb))

As a tool for dependency management and packaging in Python I choose [poetry](https://python-poetry.org/). 

All project dependencies can be founded in pyproject.toml and poetry.lock.

For linting and formatting python code were used such tools as [black](https://pypi.org/project/black/) and [flake8](https://pypi.org/project/flake8/).

For all scripts I used package [click](https://click.palletsprojects.com/en/8.1.x/) for creating CLI.



Project Organization
------------


    ├── README.md                           <- The top-level README for developers using this project.
    |
    ├── config
    │   ├── model_params.pkl                <- Parameters for CatBoostRegressor.   
    │   └── test_inst.json                  <- Data of item of beer for testing service.
    |
    ├── Docker
    |    └── Dockerfile                     <- Dockerfile for making service image.  
    |
    ├── data                    
    │   ├── beer_submission.csv             <- Final prediction for submission to Kaggle.   
    │   ├── beer_test.csv                   <- Test data from Kaggle.
    │   └── beer_train.csv                  <- Train data from Kaggle competition.  
    │            
    ├── models
    |    └── model.cbn                      <- Trained and serialized models.
    │
    ├── src                         
    |   └── beerbitterregressor
    |       ├── app
    |       |   ├── app.py                  <- Flask application
    |       |   └── test_app.py             <- Script for testing service
    |       ├── notebooks
    |       |   ├── Catboost.ipynb          <- Evaluating Catboost model
    |       |   ├── EDA.ipynb               <- Exploratory Data Analysis
    |       |   ├── OneHotEncoding.ipynb    <- Evaluating models using OneHotEncoding             
    |       |   └── TargetEncoding.ipynb    <- Evaluating models using TargetEncoding  
    |       |
    |       ├── predict.py                  <- Predict bitterness of beer
    |       ├── preprocessing.py            <- Functions for preprocessing raw data
    |       └── train.py                    <- Train a catboost model on beer dataset
    ├── .gitignore                          <- List of files git should ignore
    ├── poetry.lock                         <- File is a list of dependencies versions
    └── pyproject.toml                      <- File is a list of requirement specifiers for dependencies.


--------


## Usage

This package allows you to train model for predicting bitterness of beer, predict bitterness by using fitted model. Also you are able to run a service in docker container.

*run this and following commands in a terminal, from the root of a cloned repository*

### Preparation
1. Clone this repository to your machine.
2. Make sure Python 3.9 and [Poetry](https://python-poetry.org/docs/) are installed on your machine (I use Poetry 1.2.2).
3. Install the project dependencies:
```sh
poetry install --without dev
```
4. Install [Docker](https://www.docker.com/)

### Train
5. Run train with the following command:
```sh
poetry run train -d <path to csv with data> -m <path to save trained model> -p <path to model params>
```
### Predict**
6. Run predict with the following command:
 ```sh
poetry run predict -d <path to csv with data> -s <path to save result of prediction> -m <path of model> 
```

## Model service.

Model has been deploymented  by flask [link](https://flask.palletsprojects.com/en/2.2.x/). One way to create a WSGI server is to use gunicorn. This project was packed in a Docker container, you're able to run project on any machine.
In order to run service install docker.
Then build image
```
docker build -t beer_bitter_service -f Docker/Dockerfile .
```
Now lets run container
```
docker run -d -p 9696:9696 beer_bitter_service:latest
```
As a result, you can test service by using script src\beerbitterregressor\app\test_app.py.
```
poetry run test_app -d <path to .json with data>
```
