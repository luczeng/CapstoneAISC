import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os

# Instantiate Flask
app = Flask(__name__)

# Directory Folder to save user uploaded Patient Image Files
uploads_dir = os.path.join(app.root_path, 'static')

# Patient Log Dataframe
df = pd.DataFrame(columns=['Patient Image File','Likelihood of Disease'])

# Empty List to store user uploaded Patient Image Filenames
images = []

# Dummy Array: Test Model and Store Predictions
predictArray = np.array([[0.2, 0.8], [0.4, 0.6], [0.3, 0.7]])

# Dummy Patient Test Logs
# Obtain second index of each element (i.e. Disease Probability) from predictArray
originalLog = pd.DataFrame({
     'Patient Image File': ['p1', 'p2', 'p3'],
     'Likelihood of Disease (%)': predictArray[:,1]
     })

# Sort Disease Probality in Descending Order
updatedLog = originalLog.sort_values(by='Likelihood of Disease (%)', ascending=False)

# Dummy Model Metrics
modelMetrics = pd.DataFrame({
     'Accuracy': [1],
     'Precision': [2],
     'Recall': [3],
     'F1 Score': [4]
     })

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/upload')
def upload():
  return render_template('upload.html')


@app.route("/patientFiles", methods=['GET', 'POST'])
def patientFiles():

    for f in request.files.getlist("file"):
        filename = secure_filename(f.filename)
        f.save(os.path.join(uploads_dir, secure_filename(f.filename)))
        images.append(filename)
    print(images)

    return render_template('image.html',images=images)


@app.route('/log')
def log():

    tables = {"Original Patient Log": originalLog.to_html(classes='data', header="true"),
           "Updated Patient Log": updatedLog.to_html(classes='data', header="true"),
           "Model Metrics": modelMetrics.to_html(classes='data', header="true", index=False)
            }

    return render_template('log.html',  tables=tables)


#Run Flask env
if __name__ == "__main__":
    app.run(debug=True)
