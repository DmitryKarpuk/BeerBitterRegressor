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

Project Organization
------------


    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── beer_submission.csv       
    │   ├── beer_test_sample_submission.csv      
    │   ├── beer_test.csv      
    │   └── beer_train.csv            
    │            
    ├── models             <- Trained and serialized models.
    │
    ├── src                <- Source code for use in this project.
    |
    |
    |
    |
    |
    |
    |
--------
