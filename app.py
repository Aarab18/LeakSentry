from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pickle
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
import urllib.parse
import logging

# Setup logging for Expo debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Validate environment variables
required_env_vars = ["TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_PHONE_NUMBER", "USER_PHONE_NUMBER", "PUBLIC_URL"]
if not all(os.getenv(var) for var in required_env_vars):
    raise ValueError("Missing one or more required environment variables.")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Twilio credentials
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
USER_PHONE_NUMBER = os.getenv("USER_PHONE_NUMBER")
PUBLIC_URL = os.getenv("PUBLIC_URL")

# Initialize Twilio client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Debug mode
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"

# In-memory rate limiting
alert_timestamps = {}

# Load the trained ML model
MODEL_PATH = "model.pkl"
try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    logging.error(f"Could not find {MODEL_PATH}. Please ensure the model file exists.")
    exit(1)
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    exit(1)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided. Please adjust the sliders and try again!"}), 400

        required_fields = ["Pressure_PSI", "FlowRate_GPM", "Temperature_Cel", "Moisture_Percent", "Acoustic_dB"]
        for field in required_fields:
            if field not in data or data[field] is None:
                return jsonify({"error": f"Missing clue: {field}. Please provide all values!"}), 400

        # Input validation aligned with UI slider ranges
        inputs = {k: float(data[k]) for k in required_fields}
        validation_rules = {
            "Pressure_PSI": (0, 1000, "Pressure must be between 0 and 1000 PSI"),
            "FlowRate_GPM": (0, 100, "Flow Rate must be between 0 and 100 GPM"),
            "Temperature_Cel": (-50, 100, "Temperature must be between -50 and 100 °C"),
            "Moisture_Percent": (0, 100, "Moisture must be between 0 and 100 %"),
            "Acoustic_dB": (0, 200, "Acoustic level must be between 0 and 200 dB")
        }
        for field, (min_val, max_val, error_msg) in validation_rules.items():
            if not (min_val <= inputs[field] <= max_val):
                return jsonify({"error": error_msg}), 400

        input_features = np.array([[inputs[k] for k in required_fields]])
        prediction = model.predict(input_features)
        probability = float(model.predict_proba(input_features)[0][1])
        result = bool(prediction[0])

        # Enhanced response with severity
        severity = "Critical" if probability > 0.75 else "Moderate" if probability > 0.5 else "Low"
        response = {
            "leak": result,
            "probability": probability,
            "severity": severity if result else "None",
            "input_values": inputs,
            "message": f"{severity} leak probability detected!" if result else "No leak detected—pipes are safe!"
        }

        user_id = USER_PHONE_NUMBER
        if user_id in alert_timestamps and datetime.now() - alert_timestamps[user_id] < timedelta(minutes=5):
            response["alert"] = "Whoa, detective! Too many alerts—wait 5 minutes to investigate again."
            return jsonify(response)

        if result and not DEBUG_MODE:
            sms_message = (
                f"🚨 {severity} Leak Detected!\n"
                f"Probability: {(probability * 100):.2f}%\n"
                f"Pressure: {inputs['Pressure_PSI']} PSI\n"
                f"Flow Rate: {inputs['FlowRate_GPM']} GPM\n"
                f"Temperature: {inputs['Temperature_Cel']} °C\n"
                f"Moisture: {inputs['Moisture_Percent']} %\n"
                f"Acoustic: {inputs['Acoustic_dB']} dB"
            )
            try:
                twilio_client.messages.create(
                    body=sms_message,
                    from_=TWILIO_PHONE_NUMBER,
                    to=USER_PHONE_NUMBER
                )
                logging.info("SMS sent successfully")
            except Exception as e:
                response["sms_error"] = f"SMS failed: {str(e)}"
                logging.error(f"SMS failed: {str(e)}")

            call_message = (
                f"Alert! A {severity.lower()} leak has been detected with a probability of {probability * 100:.2f} percent. "
                "Check your SMS for details."
            )
            twiml_url = f"{PUBLIC_URL}/twiml?message={urllib.parse.quote(call_message)}"
            try:
                twilio_client.calls.create(
                    to=USER_PHONE_NUMBER,
                    from_=TWILIO_PHONE_NUMBER,
                    url=twiml_url
                )
                logging.info("Call initiated successfully")
            except Exception as e:
                response["call_error"] = f"Call failed: {str(e)}"
                logging.error(f"Call failed: {str(e)}")

            alert_timestamps[user_id] = datetime.now()
        elif DEBUG_MODE and result:
            logging.info("Debug mode: SMS and call would be sent with message: %s", sms_message)

        return jsonify(response)

    except ValueError as ve:
        return jsonify({"error": f"Invalid clue value: {str(ve)}. Check your inputs!"}), 400
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({"error": f"Mystery unsolved! Server error: {str(e)}"}), 500

@app.route("/twiml")
def twiml():
    message = request.args.get('message', 'A leak has been detected.')
    response = VoiceResponse()
    try:
        with response.say(message, voice='Polly.Amy', language='en-US') as say:
            say.pause(length=1)
            say.sentence("Please check your SMS for more details.", language='en-US')
        return str(response)
    except Exception as e:
        logging.error(f"Twiml error: {str(e)}")
        return str(VoiceResponse().say(f"Error: Could not process the message. {str(e)}")), 500

# Optional: Live prediction endpoint for real-time probability meter
@app.route("/predict/live", methods=["POST"])
def predict_live():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        required_fields = ["Pressure_PSI", "FlowRate_GPM", "Temperature_Cel", "Moisture_Percent", "Acoustic_dB"]
        inputs = {k: float(data.get(k, 0)) for k in required_fields}  # Default to 0 if missing
        input_features = np.array([[inputs[k] for k in required_fields]])
        probability = float(model.predict_proba(input_features)[0][1])

        return jsonify({"probability": probability})
    except Exception as e:
        return jsonify({"error": f"Live prediction error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)