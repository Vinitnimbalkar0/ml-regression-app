from flask import Flask,render_template,request,jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application=Flask(__name__)
app =application

#import ridge regrressor and standard scaler pickle
ridge_model = pickle.load(open('model/ridge.pkl','rb'))
standard_scaler = pickle.load(open('model/scaler.pkl','rb'))


@app.route("/")
def index():
    return render_template('home.html')

@app.route("/predict_datapoint", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            Temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))
            Classes = float(request.form.get('Classes'))
            Region = float(request.form.get('Region'))

            input_array = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
            scaled_input = standard_scaler.transform(input_array)
            prediction = ridge_model.predict(scaled_input)

            return render_template('home.html', result=round(prediction[0], 2))
        except Exception as e:
            return render_template('home.html', result=f"Error: {e}")
    
    # For GET requests, ensure result is defined
    return render_template('home.html', result=None)






if __name__=='__main__':
    app.run(host='0.0.0.0')