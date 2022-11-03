from flask import Flask, request, jsonify, Response
from catboost import CatBoostRegressor, Pool
import pandas as pd

# Define package path and madel path based on invoke source
if __name__ == "app":
    # Invokes from docker
    from preprocessing import clean_data, add_new_features

    MODEL_PATH = "model.cbn"
else:
    # Invokes directly
    from beerbitterregressor.preprocessing import clean_data, add_new_features

    MODEL_PATH = "models/model.cbn"


CAT_FEATURES = ["available", "glass"]
model = CatBoostRegressor().load_model(MODEL_PATH)

app = Flask("beer_bitter")


@app.route("/predict", methods=["POST"])
def predict() -> Response:
    beer = pd.DataFrame(request.get_json(), index=[0])
    # Prepare input data
    beer = add_new_features(clean_data(beer))
    X = beer.reset_index(drop=True)
    cat_features_idx = [list(beer.columns).index(i) for i in CAT_FEATURES]
    X_pool = Pool(X, cat_features=cat_features_idx)
    # Predict
    pred = model.predict(X_pool)
    result = {"beer bitter": round(float(pred), 2)}
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
