from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model
model = joblib.load('forestfires_model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = {
        'X': [float(request.form['X'])],
        'Y': [float(request.form['Y'])],
        'month': [int(request.form['month'])],
        'day': [int(request.form['day'])],
        'FFMC': [float(request.form['FFMC'])],
        'DMC': [float(request.form['DMC'])],
        'DC': [float(request.form['DC'])],
        'ISI': [float(request.form['ISI'])],
        'temp': [float(request.form['temp'])],
        'RH': [float(request.form['RH'])],
        'wind': [float(request.form['wind'])],
        'rain': [float(request.form['rain'])]
    }
    
    df = pd.DataFrame(input_data)
    prediction = model.predict(df)
    
    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)