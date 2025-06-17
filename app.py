from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

# Load the trained model (ensure the model is saved as 'model.pkl')
with open('model.pkl', 'rb') as file:
    Model_NB = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Serve the index.html file

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.json
        features = np.array([data['features']])  # Convert to numpy array
        prediction = predict_fraud(Model_NB, features)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

def predict_fraud(model, features):
    """
    Predicts whether the given transaction is fraud or not.
    """
    features = features[:, :4]  # Select only the first 4 features
    prediction = model.predict(features)[0]  # Get the first prediction value
    print("DEBUG: Prediction value:", prediction)
    if prediction == 1:
        return "Fraud"
    else:
        return "Not-fraud"

if __name__ == '__main__':
    app.run(debug=True)