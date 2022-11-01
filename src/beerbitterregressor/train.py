import click
import pandas as pd
from pathlib import Path
from pickle import load
from catboost import CatBoostRegressor, Pool
from .preprocessing import clean_data, add_new_features

import warnings
warnings.filterwarnings("ignore")

SEED = 42
CAT_FEATURES = ['available', 'glass']

@click.command()
@click.option(
    '-d',
    '--dataset-path',
    default='data/beer_train.csv',
    type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    '-m',
    '--model-path',
    default='models/model.cbn',
    type=click.Path(writable=True, dir_okay=False, path_type=Path)
)
@click.option(
    '-p',
    '--param-path',
    default='config/model_params.pkl',
    type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
def train(
    dataset_path: Path,
    model_path: Path,
    param_path: Path,
)-> None:
    '''
    Trains a catboost model on beer dataset with ertain parameters.
    '''

# Make model and load parameters
    model = CatBoostRegressor(verbose=1000, loss_function='RMSE')
    with open(param_path, 'rb') as f:
        model_params = load(f)
    model.set_params(**model_params)

# Load and prepare data
    data = pd.read_csv(dataset_path)
    df = clean_data(data)
    df = add_new_features(df).reset_index(drop=True)
    y_train = df['ibu'].values
    X_train = df.drop(['ibu'], axis=1).reset_index(drop=True)
    cat_features_idx = [list(X_train.columns).index(i)
                                 for i in CAT_FEATURES]
    train_pool = Pool(X_train, y_train, cat_features=cat_features_idx)

# Train and save model
    model.fit(train_pool)
    model.save_model(model_path)
    click.echo(click.style("Model was successful saved.", fg="green"))