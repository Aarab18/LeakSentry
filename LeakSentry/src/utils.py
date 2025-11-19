from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse
from email.mime.text import MIMEText
import smtplib
import os
import logging
import urllib.parse
from phonenumbers import parse, is_valid_number, format_number, PhoneNumberFormat
import numpy as np
import json
from datetime import datetime, timedelta
import certifi
import pandas as pd
import joblib
import stripe

# Twilio setup
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
USER_PHONE_NUMBER = os.getenv("USER_PHONE_NUMBER")
PUBLIC_URL = os.getenv("PUBLIC_URL")
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
TWILIO_VERIFY_SID = None  # Will be set by create_verify_service

# Stripe setup
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

# SMTP setup
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")

# Global variables
alert_timestamps = {}
demo_data = pd.read_csv('../data/demo_dataset.csv') if os.path.exists('../data/demo_dataset.csv') else None
demo_index = 0
WATER_SAVED_FILE = "../data/water_saved.json"
water_saved = float(json.load(open(WATER_SAVED_FILE, "r")).get("water_saved", 0)) if os.path.exists(WATER_SAVED_FILE) else 0
MODEL_PATH = "../model.pkl"
METRICS_PATH = "../model_metrics.pkl"
model = joblib.load(MODEL_PATH)
metrics = joblib.load(METRICS_PATH) if os.path.exists(METRICS_PATH) else None
PLANS = {
    "basic": {"name": "Basic", "price": "$0/month", "stripe_price_id": None},
    "pro": {"name": "Pro", "price": "$9.99/month", "stripe_price_id": "price_1RCDwPFPjZ4plTBWdNI99otk"},
    "enterprise": {"name": "Enterprise", "price": "$29.99/month", "stripe_price_id": "price_1RCDxHFPjZ4plTBWXlZKoqw6"}
}
CHATBOT_RESPONSES = {
    "hello|hi|hey": "Hi there! I'm LeakSentry's assistant. Ask me about leak detection, pricing, or system status!",
    "work|how does|what is leaksentry": "LeakSentry uses AI to monitor pressure, flow, temperature, moisture, and acoustic data for real-time leak detection. Try the Detect page to test it out!",
    "status|system|check": "All systems are running smoothly! Want to check for leaks? Visit the Detect page.",
    "leak|help|problem": "Suspect a leak? Head to the Detect page to input sensor data. For urgent issues, call +91 7068460882.",
    "price|pricing|plan|cost": "We offer Basic ($0/month), Pro ($9.99/month), and Enterprise ($29.99/month) plans. See the Plans page for details.",
    "water saved|conservation": f"LeakSentry has helped save {water_saved} liters of water so far! Run a detection to contribute more.",
    "support|contact|help me": "For support, email leaksentry@gmail.com or call +91 7068460882. I'm here for quick questions too!",
    "team|who made": "LeakSentry was built by Aarab, Kushagra, and Aayush. Check the Team page for more about them!",
    "feedback|review": "We'd love your feedback! Visit the Feedback page to share your thoughts.",
    "history|past detections": "You can view past leak detections on the History page.",
    "default": "Hmm, I'm not sure about that one. Try asking about leaks, pricing, or how LeakSentry works!"
}
FEEDBACK_FILE = "../data/feedback.json"
HISTORY_FILE = "../data/detection_history.json"
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"

# SSL fix
os.environ['SSL_CERT_FILE'] = certifi.where()

# Utility functions
def create_verify_service():
    try:
        verify_service = twilio_client.verify.v2.services.create(friendly_name="LeakSentry")
        return verify_service.sid
    except Exception as e:
        logging.warning(f"Could not create Verify service: {str(e)}. Trying to reuse existing...")
        services = twilio_client.verify.v2.services.list()
        leak_service = next((s for s in services if s.friendly_name == "LeakSentry"), None)
        if leak_service:
            return leak_service.sid
        logging.error("No existing Verify service found with friendly_name='LeakSentry'.")
        raise

def send_email(to_email, subject, body):
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = SMTP_USERNAME
        msg['To'] = to_email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
        logging.info(f"Email sent successfully to {to_email}")
        return True
    except Exception as e:
        logging.error(f"Failed to send email to {to_email}: {str(e)}")
        return False

def twiml_response(message):
    response = VoiceResponse()
    try:
        with response.say(message, voice='Polly.Amy', language='en-US') as say:
            say.pause(length=1)
            say.sentence("Please check your SMS for more details.", language='en-US')
        return str(response)
    except Exception as e:
        logging.error(f"Twiml error: {str(e)}")
        return str(VoiceResponse().say(f"Error: {str(e)}"))

def validate_phone_number(phone, country="IN"):
    try:
        parsed_phone = parse(phone, country)
        if not is_valid_number(parsed_phone):
            return False, "Invalid phone number."
        normalized_phone = format_number(parsed_phone, PhoneNumberFormat.E164)
        if normalized_phone.startswith('+1'):
            return False, "Phone numbers with country code +1 are not supported. Please use +91."
        return True, normalized_phone
    except Exception as e:
        logging.error(f"Phone parsing failed: {str(e)}")
        return False, "Invalid phone number format."