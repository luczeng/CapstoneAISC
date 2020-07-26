# -*- coding: utf-8 -*-
"""
Test Flask API.
"""

from Capstone.configs.read_config import parse_config
import argparse
import pandas as pd
import mlflow.pyfunc
import json
from flask import Flask, render_template, redirect, url_for, request, jsonify
import pydicom

def parse_args():

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-c", "--config_path", type=str, required=True)
    parser.add_argument("-i", "--input_image", type=str, required=True)
    args = parser.parse_args()

    return args

app = Flask(__name__)

# DataFrames
originalLog = pd.DataFrame(
    {
        "Order": ["1", "2", "3"],
        "Patient ID": [123, 456, 789],
        "Date of Test": ["July 02, 2020", "July 09, 2020", "July 15, 2020"],
        "Likelihood of Disease (%)": [80, 95, 93],
    }
)

updatedLog = pd.DataFrame(
    {
        "Order": ["1", "2", "3"],
        "Patient ID": [456, 789, 123],
        "Date of Test": ["July 09, 2020", "July 15, 2020", "July 02, 2020"],
        "Likelihood of Disease (%)": [95, 93, 80],
    }
)


@app.route("/")

# Informoation endpoint
@app.route("/home")
def home():
    return render_template("home.html")


# Example image endpoint
@app.route("/image")
def image():
    return render_template("image.html")


# Example endpoint
@app.route("/about")
def about():
    return render_template("about.html")

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():

    # retrieve request
    member = request.data
    folder_path = member.decode('utf-8')

    # Load image
    data = dataset = pydicom.dcmread(folder_path)
    img = pd.DataFrame(data.pixel_array)

    # Format the request to a dataframe
    # inf_df = pd.DataFrame(req['data'])
    pred = model.predict(img).tolist()
    print(pred)

	# Return prediction as reponse
    return jsonify(pred)

# Run Flask env
if __name__ == "__main__":
    # Configs
    args = parse_args()
    cfg = parse_config(args.config_path)

    # Load model at app startup
    model = mlflow.pyfunc.load_model(cfg.mlflow_pyfunc_model_path)

    app.run(host='0.0.0.0', port=5000, debug=True)
