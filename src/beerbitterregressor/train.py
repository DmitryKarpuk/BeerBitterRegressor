import click
import pandas as pd
from pathlib import Path
from pickle import load
from catboost import CatBoostRegressor, Pool
from .preprocessing import clean_data, add_new_features

import warnings

warnings.filterwarnings("ignore")

SEED = 42
CAT_FEATURES = ["available", "glass"]


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/beer_train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "-m",
    "--model-path",
    default="models/model.cbn",
    type=click.Path(writable=True, dir_okay=False, path_type=Path),
)
@click.option(
    "-p",
    "--param-path",
    default="config/model_params.pkl",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
def train(
    dataset_path: Path,
    model_path: Path,
    param_path: Path,
) -> None:
    """
    Train a catboost model on beer dataset with сertain parameters.
    Save trained model to .cbn file.
    
    Args:
        dataset_path: Path of file with data of beer.
        model_path: Path of model file.
        param_path: Path of model parameters.
 
    Return:
        None
    """

    # Make model and load parameters
    model = CatBoostRegressor(verbose=1000, loss_function="RMSE")
    with open(param_path, "rb") as f:
        model_params = load(f)
    model.set_params(**model_params)

    # Load and prepare data
    data = pd.read_csv(dataset_path, index_col=["id"])
    df = clean_data(data)
    df = add_new_features(df).reset_index(drop=True)
    train_features = [x for x in df.columns if x != "ibu"]
    y_train = df["ibu"].values
    X_train = df[train_features].reset_index(drop=True)
    cat_features_idx = [train_features.index(i) for i in CAT_FEATURES]
    train_pool = Pool(X_train, y_train, cat_features=cat_features_idx)

    # Train and save model
    model.fit(train_pool)
    model.save_model(model_path)
    click.echo(click.style("Model was successful saved.", fg="green"))
