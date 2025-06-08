from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model
import joblib
model = joblib.load('model.pkl')


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    int_features = []
    for x in request.form.values():
      if x.lower() == 'true':
          int_features.append(1)
      elif x.lower() == 'false':
          int_features.append(0)
      else:
          int_features.append(float(x))

    final_features = [np.array(int_features)]
    
    # Make prediction
    prediction = model.predict(final_features)
    output = 'YES' if prediction[0] == 1 else 'NO'

    return render_template('index.html', prediction_text='Prediction: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)