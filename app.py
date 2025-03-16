from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the trained ML model
MODEL_PATH = "model.pkl"  # Ensure you have a trained model saved as model.pkl
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Convert input data to numpy array for model prediction
        input_features = np.array([
            [
                float(data["Pressure_PSI"]),
                float(data["FlowRate_GPM"]),
                float(data["Temperature_Cel"]),
                float(data["Moisture_Percent"]),
                float(data["Acoustic_dB"]),
            ]
        ])

        # Make prediction (assuming binary classification: 1 = Leak, 0 = No Leak)
        prediction = model.predict(input_features)
        result = bool(prediction[0])  # Convert NumPy bool to Python bool

        # Return the result as JSON
        return jsonify({"leak": result})

    except Exception as e:
        return jsonify({"error": str(e)})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, port=5000)
