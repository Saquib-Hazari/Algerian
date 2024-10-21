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
def hello_world():
      return render_template('index.html')

if __name__ == "__main__":
      app.run(host="0.0.0.0")