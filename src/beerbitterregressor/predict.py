import click
import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostRegressor, Pool
from .preprocessing import clean_data, add_new_features


CAT_FEATURES = ["available", "glass"]


def _predict(X: Pool, model_path: Path) -> np.ndarray:
    '''
    Predict bitterness of beer by using pretrained CatBoost model.

    Args:
        X: Pool with test data
        model_path: Path of model file.
    
    Return:
        Numpy array of prediction.
    '''
    model = CatBoostRegressor()
    model.load_model(model_path)
    pred = model.predict(X)
    return pred


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/beer_test.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "-s",
    "--submission-path",
    default="data/beer_submission.csv",
    type=click.Path(writable=True, dir_okay=False, path_type=Path),
)
@click.option(
    "-m",
    "--model-path",
    default="models/model.cbn",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
def predict(
    dataset_path: Path, submission_path: Path, model_path: Path
) -> None:
    '''
    Predict bitterness of beer by using pretrained CatBoost model.
    Save prediction to csv file.
    File format corresponds to Kaggle submission file format.  

    Args:
        dataset_path: Path of file with data of beer.
        submission_path: Path of submission file.
    
    Return:
        None.
    '''
    # Load data and prepare data
    data = pd.read_csv(dataset_path, index_col=["id"])
    df = clean_data(data)
    df = add_new_features(df)
    X = df.reset_index(drop=True)
    cat_features_idx = [list(df.columns).index(i) for i in CAT_FEATURES]
    X_pool = Pool(X, cat_features=cat_features_idx)
    # Predict on a Pandas DataFrame.
    pred = _predict(X_pool, model_path)
    pred = np.round_(pred, 2)
    submission = pd.DataFrame({"ibu": pred}, index=df.index)
    submission.to_csv(submission_path)
    click.echo(
        click.style(f"Submission is saved to {submission_path}", fg="green")
    )
