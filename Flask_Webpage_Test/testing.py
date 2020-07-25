# -*- coding: utf-8 -*-
"""
Test Flask API.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template, redirect, url_for, request


app = Flask(__name__)

# DataFrames
originalLog = pd.DataFrame({
     'Order': ['1', '2', '3'],
     'Patient ID': [123, 456, 789],
     'Date of Test': ['July 02, 2020', 'July 09, 2020', 'July 15, 2020'],
     'Likelihood of Disease (%)': [80, 95, 93]
     })

updatedLog = pd.DataFrame({
     'Order': ['1', '2', '3'],
     'Patient ID': [456, 789, 123],
     'Date of Test': ['July 09, 2020', 'July 15, 2020', 'July 02, 2020'],
     'Likelihood of Disease (%)': [95, 93, 80]
     })

'''
# Use this to get HTML Code and copy HTML Code to Template Script

# render dataframe as html
html = originalLog.to_html()
html = updatedLog.to_html()

print(html)

'''

@app.route('/')

@app.route('/home')
def home():
    return render_template('home.html')
    
@app.route('/image')
def image():
    return render_template('image.html')

@app.route('/about')
def about():
    return render_template('about.html')

#Run Flask env 
if __name__ == "__main__":
    app.run(debug=True)