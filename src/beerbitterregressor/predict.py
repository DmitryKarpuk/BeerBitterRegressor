import click
import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostRegressor, Pool
from .preprocessing import clean_data, add_new_features


CAT_FEATURES = ["available", "glass"]


def _predict(X: Pool, model_path: Path) -> np.ndarray:
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
    # Load data and prepare data
    data = pd.read_csv(dataset_path, index_col=["id"])
    df = clean_data(data)
    df = add_new_features(df)
    X = df.reset_index(drop=True)
    cat_features_idx = [list(df.columns).index(i) for i in CAT_FEATURES]
    X_pool = Pool(X, cat_features=cat_features_idx)
    # Predict on a Pandas DataFrame.
    pred = _predict(X_pool, model_path)
    submission = pd.DataFrame({"ibu": pred}, index=df.index)
    submission.to_csv(submission_path)
    click.echo(
        click.style(f"Submission is saved to {submission_path}", fg="green")
    )
