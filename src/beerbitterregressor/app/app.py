from flask import Flask, request, jsonify, Response
from catboost import CatBoostRegressor, Pool
import pandas as pd

from ..preprocessing import clean_data, add_new_features

CAT_FEATURES = ["available", "glass"]
MODEL_PATH = "models/model.cbn"

model = CatBoostRegressor().load_model(MODEL_PATH)

app = Flask('beer_bitter')


@app.route('/predict', methods=['POST'])
def predict() -> Response:
    beer = request.get_json()
    # Prepare input data
    beer = add_new_features(clean_data(beer))
    X = beer.reset_index(drop=True)
    cat_features_idx = [list(beer.columns).index(i) for i in CAT_FEATURES]
    X_pool = Pool(X, cat_features=cat_features_idx)
    # Predict
    pred = round(model.predict(X_pool), 2)
    result = {"beer bitter": float(pred)}
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
