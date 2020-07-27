# -*- coding: utf-8 -*-
"""
Test Flask API.
"""

import numpy as np
import pandas as pd
from flask import Flask, render_template, request
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
updatedLog = pd.DataFrame({
     'Patient Image File': ['p1', 'p2', 'p3'],
     'Likelihood of Disease (%)': -np.sort(-predictArray[:,1])
     })

'''
# Use this to get HTML Code and copy HTML Code to Template Script
# render dataframe as html
html = originalLog.to_html()
html = updatedLog.to_html()
print(html)
'''

@app.route('/')
def home():
    return render_template('home.html')
  
@app.route("/uploadComplete", methods=['GET', 'POST'])
def uploadComplete():

    for f in request.files.getlist("file"):
        filename = secure_filename(f.filename)
        f.save(os.path.join(uploads_dir, secure_filename(f.filename)))
        images.append(filename) 
    print(images)
        
    return render_template('image.html',images = images)
    
@app.route('/about')
def about():
    
    # TO BE COMPLETED
    # Update Log DataFrame with Uploaded Image Filenames & Model Preductions 
    print(df)
    
    return render_template('about.html')

#Run Flask env 
if __name__ == "__main__":
    app.run(debug=True)