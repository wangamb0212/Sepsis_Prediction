from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

model = joblib.load('random_forest_model.pkl')

@app.route('/')
def home():
    
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_features = [
        float(request.form['Hour']),  
        float(request.form['HR']),
        float(request.form['O2Sat']),
        float(request.form['Temp']),
        float(request.form['Resp']),
        float(request.form['BaseExcess']),  
        float(request.form['SaO2']), 
        float(request.form['AST']), 
        float(request.form['BUN']),
        float(request.form['Alkalinephos']),
        float(request.form['Calcium']),
        float(request.form['Chloride']),
        float(request.form['Bilirubin_direct']), 
        float(request.form['Glucose']),
        float(request.form['Lactate']),  
        float(request.form['Magnesium']),
        float(request.form['Potassium']),
        float(request.form['PTT']),
        float(request.form['WBC']),
        float(request.form['Fibrinogen']),
        float(request.form['Platelets']),
        float(request.form['Age']),  
        float(request.form.get('Unit1', 0)),  # whether it's MICU or SICU
    ]
    
    # Make a prediction
    prediction = model.predict([input_features])

    # Format the prediction for presentation
    output = "Class: " + str(prediction[0])

    return render_template('index.html', prediction_text=f'Prediction: {output}')

if __name__ == "__main__":
    app.run(debug=True)
