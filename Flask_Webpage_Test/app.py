# -*- coding: utf-8 -*-
"""
Test Flask API.
"""

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from Capstone.configs.read_config import parse_config
import argparse
import mlflow.pyfunc
from pathlib import Path
from Capstone.io.io import load_dicom


def parse_args():

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-c", "--config_path", type=str, required=True)
    args = parser.parse_args()

    return args


# Instantiate Flask
app = Flask(__name__)

# Directory Folder to save user uploaded Patient Image Files
uploads_dir = os.path.join(app.root_path, "static")

# Patient Log Dataframe
df = pd.DataFrame(columns=["Patient Image File", "Likelihood of Disease"])

# Empty List to store user uploaded Patient Image Filenames
images = []

# Dummy Array: Test Model and Store Predictions
predictArray = np.array([[0.2, 0.8], [0.4, 0.6], [0.3, 0.7]])

# Dummy Patient Test Logs
# Obtain second index of each element (i.e. Disease Probability) from predictArray
originalLog = pd.DataFrame({"Patient Image File": ["p1", "p2", "p3"], "Likelihood of Disease (%)": predictArray[:, 1]})

# Sort Disease Probality in Descending Order
updatedLog = pd.DataFrame(
    {"Patient Image File": ["p1", "p2", "p3"], "Likelihood of Disease (%)": -np.sort(-predictArray[:, 1])}
)

"""
# Use this to get HTML Code and copy HTML Code to Template Script
# render dataframe as html
html = originalLog.to_html()
html = updatedLog.to_html()
print(html)
"""


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

    # prediction = np.asarray(predictions)
    print(predictions)

    # Return prediction as reponse
    updatedLog = pd.DataFrame(
        {"Patient Image File": ["p1", "p2"], "Likelihood of Disease (%)": [predictions[0][1], predictions[1][1]]}
    )
    # html_file = pd.DataFrame(predictions).to_html()
    html_file = pd.DataFrame(updatedLog).to_html()

    with open('Flask_Webpage_Test/templates/predict.html','w') as f:
        f.write(html_file)

    return render_template('predict.html')

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
    cfg = parse_config(args.config_path)

    # Load model at app startup
    model = mlflow.pyfunc.load_model(cfg.mlflow_pyfunc_model_path)

    app.run(host="0.0.0.0", port=5000, debug=True)

# # -*- coding: utf-8 -*-
# """
# Test Flask API.
# """

# from Capstone.configs.read_config import parse_config
# import argparse
# import pandas as pd
# import mlflow.pyfunc
# import json
# from flask import Flask, render_template, redirect, url_for, request, jsonify
# from pathlib import Path
# from Capstone.io.io import load_dicom


# app = Flask(__name__)

# # DataFrames
# originalLog = pd.DataFrame(
# {
# "Order": ["1", "2", "3"],
# "Patient ID": [123, 456, 789],
# "Date of Test": ["July 02, 2020", "July 09, 2020", "July 15, 2020"],
# "Likelihood of Disease (%)": [80, 95, 93],
# }
# )

# updatedLog = pd.DataFrame(
# {
# "Order": ["1", "2", "3"],
# "Patient ID": [456, 789, 123],
# "Date of Test": ["July 09, 2020", "July 15, 2020", "July 02, 2020"],
# "Likelihood of Disease (%)": [95, 93, 80],
# }
# )


# @app.route("/")

# # Informoation endpoint
# @app.route("/home")
# def home():
# return render_template("home.html")


# # Example image endpoint
# @app.route("/image")
# def image():
# return render_template("image.html")


# # Example endpoint
# @app.route("/about")
# def about():
# return render_template("about.html")
