import requests
import json
from pathlib import Path
import click

URL = "http://127.0.0.1:9696/predict"


@click.command(help="Script with request for testing model app.")
@click.option(
    "-p",
    "--data-path",
    default="config/test_inst.json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
def predict_req(data_path: Path) -> None:
    """
    Test machine learning service on url using one item of beer.
    
    Args:
        data_path: Path of file with one item of beer.
 
    Return:
        Bitterness of current beer. 
    """
    with open(data_path) as f:
        beer = json.load(f)
    response = requests.post(URL, json=beer)
    result = response.json()
    print(result)
