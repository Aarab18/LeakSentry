# from flask import Flask, request, jsonify, render_template
# import numpy as np
# import pickle

# # Initialize Flask app
# app = Flask(__name__)

# # Load the trained ML model
# MODEL_PATH = "model.pkl"  # Ensure you have a trained model saved as model.pkl
# with open(MODEL_PATH, "rb") as file:
#     model = pickle.load(file)

# @app.route("/")
# def home():
#     return render_template('index.html')

# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         # Get JSON data from the request
#         data = request.get_json()

#         # Convert input data to numpy array for model prediction
#         input_features = np.array([
#             [
#                 float(data["Pressure_PSI"]),
#                 float(data["FlowRate_GPM"]),
#                 float(data["Temperature_Cel"]),
#                 float(data["Moisture_Percent"]),
#                 float(data["Acoustic_dB"]),
#             ]
#         ])

#         # Make prediction (assuming binary classification: 1 = Leak, 0 = No Leak)
#         prediction = model.predict(input_features)
#         result = bool(prediction[0])  # Convert NumPy bool to Python bool

#         # Return the result as JSON
#         return jsonify({"leak": result})

#     except Exception as e:
#         return jsonify({"error": str(e)})

# # Run the Flask app
# if __name__ == "__main__":
#     app.run(debug=True, port=5000)


from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pickle

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS to allow requests from frontend

# Load the trained ML model
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
    # Assuming you have an index.html in a 'templates' folder
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Validate and extract all required fields
        required_fields = ["Pressure_PSI", "FlowRate_GPM", "Temperature_Cel", "Moisture_Percent", "Acoustic_dB"]
        for field in required_fields:
            if field not in data or data[field] is None:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # Convert input data to numpy array
        input_features = np.array([[
            float(data["Pressure_PSI"]),
            float(data["FlowRate_GPM"]),
            float(data["Temperature_Cel"]),
            float(data["Moisture_Percent"]),
            float(data["Acoustic_dB"]),
        ]])

        # Make prediction
        prediction = model.predict(input_features)
        result = bool(prediction[0])  # Convert to Python bool (0 = No Leak, 1 = Leak)

        # Return result as JSON
        return jsonify({"leak": result})

    except ValueError as ve:
        return jsonify({"error": f"Invalid input value: {str(ve)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)