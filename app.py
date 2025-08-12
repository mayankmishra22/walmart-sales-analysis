import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle
import datetime as dt
import calendar
import os
import urllib.request

app = Flask(__name__)

# Model paths and URL
MODEL_PATH = "rf_model.pkl"
MODEL_URL = "https://github.com/mayankmishra22/walmart-sales-analysis/releases/download/v1.0-model/rf_model.pkl"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from GitHub Releases...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded successfully.")

print("Loading model into memory...")
with open(MODEL_PATH, 'rb') as f:
    loaded_model = pickle.load(f)
print("Model loaded successfully.")

# Load CSV data
fet = pd.read_csv('merged_data.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    store = request.form.get('store')
    dept = request.form.get('dept')
    size = request.form.get('size')
    temp = request.form.get('temp')
    isHoliday = request.form['isHolidayRadio']
    date = request.form.get('date')

    d = dt.datetime.strptime(date, '%Y-%m-%d')
    month = d.month
    year = d.year
    month_name = calendar.month_name[month]

    print("year =", type(year))
    print("year val =", year, type(year), month)

    X_test = pd.DataFrame({
        'Store': [store],
        'Dept': [dept],
        'Size': [size],
        'IsHoliday': [isHoliday],
        'CPI': [212],
        'Temperature': [temp],
        'Type_A': [0],
        'Type_B': [0],
        'Type_C': [1],
        'month': [month],
        'Year': [year]
    })[["Store", "Dept", "Size", "IsHoliday", "CPI", "Temperature",
        "Type_A", "Type_B", "Type_C", "month", "Year"]]

    print("X_test =", X_test.head())
    print("type of X_test =", type(X_test))
    print("predict =", store, dept, date, isHoliday)

    y_pred = loaded_model.predict(X_test)
    output = round(y_pred[0], 2)
    print("predicted =", output)

    return render_template('index.html', output=output, store=store, dept=dept,
                           month_name=month_name, year=year)

if __name__ == "__main__":
    app.run(debug=False)
