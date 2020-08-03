# -*- coding: utf-8 -*-
"""
Test Flask API.
"""

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import argparse
import mlflow.pyfunc
from pathlib import Path

# from Capstone.io.io import load_dicom


def parse_args():

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-m", "--mlflow_pyfunc_model_path", type=str, required=True)
    args = parser.parse_args()

    return args


# Instantiate Flask
app = Flask(__name__)

# Directory Folder to save user uploaded Patient Image Files
uploads_dir = os.path.join(app.root_path, "static")


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/uploadComplete", methods=["GET", "POST"])
def uploadComplete():

    for f in request.files.getlist("file"):
        filename = secure_filename(f.filename)
        f.save(os.path.join(uploads_dir, secure_filename(f.filename)))
        images.append(filename)
    print(images)

    return render_template("image.html", images=images)


@app.route("/about")
def about():

    # TO BE COMPLETED
    # Update Log DataFrame with Uploaded Image Filenames & Model Preductions
    print(df)

    return render_template("about.html")


@app.route("/predict")
def predict():
    """
        Loads request from disk and then launch prediction
    """

    # Create panda dataframe
    predictions = []
    for fp in Path(uploads_dir).iterdir():

        # Load image
        img = load_dicom(fp)

        # Format the request to a dataframe
        img_pf = pd.DataFrame(img)
        pred_arr = model.predict(img_pf)

        # Append predictions
        predictions.append(pred_arr[0].tolist())

    # Return prediction as reponse
    patients_list = [f"patient{idx}" for idx in range(len(predictions))]
    probabilities = [predictions[idx][1] for idx in range(len(predictions))]
    updatedLog = pd.DataFrame({"Patient Image File": patients_list, "Likelihood of Disease (%)": probabilities})
    # html_file = pd.DataFrame(predictions).to_html()
    html_file = pd.DataFrame(updatedLog).to_html()

    with open("Flask_Webpage_Test/templates/predict.html", "w") as f:
        f.write(html_file)

    return render_template("predict.html")


# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict_api():
    """
        Loads request from disk and then launch prediction
    """

    # retrieve request
    member = request.data
    folder_path = member.decode("utf-8")

    # Create panda dataframe
    predictions = []
    for fp in Path(folder_path).iterdir():

        # Load image
        img = load_dicom(fp)

        # Format the request to a dataframe
        img_pf = pd.DataFrame(img)
        pred_arr = model.predict(img_pf)

        # Append predictions
        predictions.append(pred_arr[0].tolist())

    # Return prediction as reponse
    return jsonify(predictions)


# Run Flask env
if __name__ == "__main__":
    # Configs
    args = parse_args()

    # Load model at app startup
    model = mlflow.pyfunc.load_model(args.mlflow_pyfunc_model_path)

    app.run(host="0.0.0.0", port=5000, debug=True)
