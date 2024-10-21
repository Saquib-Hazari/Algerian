from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

app = Flask(__name__)

# Import the Model ridge and scaler
ridge_model = pickle.load(open('Model/rd.pkl', 'rb'))
scaler_model = pickle.load(open('Model/scaler.pkl', 'rb'))

@app.route("/")
def index():
      return render_template('index.html')

@app.route("/predictdata", methods=['GET', 'POST'])
def predict_datapoint():
      if request.method == "POST":
            temperature = float(request.form.get('temperature'))
            rh = float(request.form.get('rh'))
            ws = float(request.form.get('ws'))
            rain = float(request.form.get('rain'))
            ffmc = float(request.form.get('ffmc'))
            dmc = float(request.form.get('dmc'))
            isi = float(request.form.get('isi'))
            classes = float(request.form.get('classes'))
            region = float(request.form.get('region'))

            new_data_scale = scaler_model.transform([[temperature, rh, ws, rain, ffmc, dmc, isi, classes, region]])

            result = ridge_model.predict(new_data_scale)
            return render_template('home.html', results=result[0])
      else:
            return render_template('home.html')

if __name__ == "__main__":
      app.run(debug=True)