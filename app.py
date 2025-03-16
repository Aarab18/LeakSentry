#importing the required libraries
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pickle

#initializing the Flask app
app = Flask(__name__)

#enabling CORS to allow requests from frontend
CORS(app) 

#loading the trained ML model saved as .pkl file
MODEL_PATH = "model.pkl"
try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    print(f"Error: Could not find {MODEL_PATH}. Please ensure the model file exists.")
    exit(1)
except Exception as e:
    print(f"Error loading model: {str(e)}")
    exit(1)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    try:
        #getting JSON data from the request
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        #validating and extracting all required fields
        required_fields = ["Pressure_PSI", "FlowRate_GPM", "Temperature_Cel", "Moisture_Percent", "Acoustic_dB"]
        for field in required_fields:
            if field not in data or data[field] is None:
                return jsonify({"error": f"Missing field: {field}"}), 400

        #converting input data to numpy array
        input_features = np.array([[
            float(data["Pressure_PSI"]),
            float(data["FlowRate_GPM"]),
            float(data["Temperature_Cel"]),
            float(data["Moisture_Percent"]),
            float(data["Acoustic_dB"]),
        ]])

        #making prediction
        prediction = model.predict(input_features)

        #converting to boolean (0 = No Leak, 1 = Leak)
        result = bool(prediction[0])

        #returnng result as JSON
        return jsonify({"leak": result})

    except ValueError as ve:
        return jsonify({"error": f"Invalid input value: {str(ve)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)