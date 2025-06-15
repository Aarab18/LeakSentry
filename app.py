from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session, send_file
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, TelField, DateField
from wtforms.validators import DataRequired, Email, Length, EqualTo
from flask_bcrypt import Bcrypt
import numpy as np
import pickle
import joblib
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
import urllib.parse
import logging
import stripe
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.linear_model import LinearRegression
import json
from phonenumbers import parse, is_valid_number, format_number, PhoneNumberFormat
import smtplib
from email.mime.text import MIMEText
import certifi
import qrcode
import io
from pyngrok import ngrok
import subprocess
import requests

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Validate environment variables
required_env_vars = [
    "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_PHONE_NUMBER",
    "USER_PHONE_NUMBER", "STRIPE_SECRET_KEY", "STRIPE_PUBLISHABLE_KEY",
    "SMTP_SERVER", "SMTP_PORT", "SMTP_USERNAME", "SMTP_PASSWORD"
]
if not all(os.getenv(var) for var in required_env_vars):
    raise ValueError("Missing one or more required environment variables.")

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY", os.urandom(24))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SESSION_PERMANENT'] = False  # Sessions expire on browser close
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)  # Fallback timeout
app.config['SESSION_COOKIE_SECURE'] = True  # Requires HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True  # Prevent JS access
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # CSRF protection
CORS(app)

# Initialize extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Stripe setup
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

# Twilio credentials
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
USER_PHONE_NUMBER = os.getenv("USER_PHONE_NUMBER")

# SMTP credentials
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")

# Admin contact details
ADMIN_PHONE = os.getenv("ADMIN_PHONE")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL")

# Initialize Twilio client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Fix SSL cert verification issues
os.environ['SSL_CERT_FILE'] = certifi.where()

# Create Twilio Verify service or reuse existing one
try:
    verify_service = twilio_client.verify.v2.services.create(friendly_name="LeakSentry")
    TWILIO_VERIFY_SID = verify_service.sid
except Exception as e:
    logging.warning(f"Could not create Verify service: {str(e)}. Trying to reuse existing...")
    try:
        services = twilio_client.verify.v2.services.list()
        leak_service = next((s for s in services if s.friendly_name == "LeakSentry"), None)
        if leak_service:
            TWILIO_VERIFY_SID = leak_service.sid
        else:
            logging.error("No existing Verify service found with friendly_name='LeakSentry'.")
            raise
    except Exception as inner_e:
        logging.error(f"Failed to list existing services: {str(inner_e)}")
        raise

# Debug mode
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"

# Load or initialize water saved from file
WATER_SAVED_FILE = "data/water_saved.json"
if os.path.exists(WATER_SAVED_FILE):
    with open(WATER_SAVED_FILE, "r") as f:
        water_saved = float(json.load(f).get("water_saved", 0))
else:
    water_saved = 0

# In-memory rate limiting and demo data
alert_timestamps = {}
demo_data = None
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    demo_csv_path = os.path.join(script_dir, 'demo_dataset.csv')
    demo_data = pd.read_csv(demo_csv_path)
except FileNotFoundError:
    logging.error(f"Could not find 'demo_dataset.csv' at {demo_csv_path}. Demo data will be unavailable.")
    demo_data = None
except Exception as e:
    logging.error(f"Error loading demo dataset: {str(e)}")
    demo_data = None

demo_index = 0

# Load the trained ML model and metrics
MODEL_PATH = "model.pkl"
METRICS_PATH = "model_metrics.pkl"
try:
    model = joblib.load(MODEL_PATH)
    metrics = joblib.load(METRICS_PATH) if os.path.exists(METRICS_PATH) else None
    logging.info("Model and metrics loaded successfully")
except FileNotFoundError:
    logging.error(f"Could not find {MODEL_PATH}. Please ensure the model file exists.")
    exit(1)
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    exit(1)

# Load or train maintenance prediction model
MAINTENANCE_MODEL_PATH = "maintenance_model.pkl"
try:
    maintenance_model = joblib.load(MAINTENANCE_MODEL_PATH)
except FileNotFoundError:
    try:
        with open("data/detection_history.json", "r") as f:
            history = json.load(f)
        X = np.array([[item['probability']] for item in history if item.get('leak')])
        y = np.array([1 if item['severity'] == 'Critical' else 0 for item in history if item.get('leak')])
        if len(X) > 0 and len(y) > 0:
            maintenance_model = LinearRegression().fit(X, y)
            joblib.dump(maintenance_model, MAINTENANCE_MODEL_PATH)
        else:
            maintenance_model = LinearRegression()
            joblib.dump(maintenance_model, MAINTENANCE_MODEL_PATH)
    except Exception as e:
        logging.error(f"Error training maintenance model: {str(e)}")
        maintenance_model = LinearRegression()
        joblib.dump(maintenance_model, MAINTENANCE_MODEL_PATH)

# Subscription plans
PLANS = {
    "basic": {"name": "Basic", "price": "$0/month", "stripe_price_id": None},
    "pro": {"name": "Pro", "price": "$9.99/month", "stripe_price_id": "price_1RCDwPFPjZ4plTBWdNI99otk"},
    "enterprise": {"name": "Enterprise", "price": "$29.99/month", "stripe_price_id": "price_1RCDxHFPjZ4plTBWXlZKoqw6"}
}

# Enhanced chatbot responses
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

# Feedback and history storage paths
FEEDBACK_FILE = "data/feedback.json"
HISTORY_FILE = "data/detection_history.json"

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# Initialize feedback and history files if they don't exist
if not os.path.exists(FEEDBACK_FILE):
    with open(FEEDBACK_FILE, "w") as f:
        json.dump([], f)

if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w") as f:
        json.dump([], f)

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    dob = db.Column(db.String(10), nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

# Login form
class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

# Registration form
class RegisterForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    phone = TelField('Phone Number', validators=[DataRequired()])
    name = StringField('Full Name', validators=[DataRequired()])
    dob = DateField('Date of Birth', validators=[DataRequired()], format='%Y-%m-%d')
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Send OTP')

# OTP verification form
class OTPForm(FlaskForm):
    otp = StringField('OTP', validators=[DataRequired(), Length(min=4, max=6)])
    submit = SubmitField('Verify OTP')

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Session timeout middleware
@app.before_request
def before_request():
    session.permanent = False  # Ensure non-permanent sessions
    if current_user.is_authenticated:
        # Track last activity time
        if 'last_activity' not in session:
            session['last_activity'] = datetime.utcnow().isoformat()
        last_activity = datetime.fromisoformat(session['last_activity'])
        # Check if session has timed out (30 minutes)
        if datetime.utcnow() - last_activity > timedelta(minutes=30):
            logout_user()
            session.clear()
            flash('Your session has timed out. Please log in again.', 'error')
            return redirect(url_for('login'))
        # Update last activity time
        session['last_activity'] = datetime.utcnow().isoformat()

# Email sending function
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

@app.route("/")
def home():
    return render_template('home.html', water_saved=water_saved)

@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password_hash, form.password.data):
            login_user(user)
            session.permanent = False  # Explicitly non-permanent session
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        flash('Invalid email or password.', 'error')
    return render_template('login.html', form=form)

@app.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RegisterForm()
    if form.validate_on_submit():
        if User.query.filter_by(email=form.email.data).first():
            flash('Email already registered.', 'error')
            return render_template('register.html', form=form)

        try:
            parsed_phone = parse(form.phone.data, "IN")
            if not is_valid_number(parsed_phone):
                flash('Invalid phone number.', 'error')
                return render_template('register.html', form=form)
            normalized_phone = format_number(parsed_phone, PhoneNumberFormat.E164)
            if normalized_phone.startswith('+1'):
                flash('Phone numbers with country code +1 are not supported. Please use +91.', 'error')
                return render_template('register.html', form=form)
        except Exception as e:
            logging.error(f"Register: Phone parsing failed: {str(e)}")
            flash('Invalid phone number format.', 'error')
            return render_template('register.html', form=form)

        session.clear()
        session['register_data'] = {
            'email': form.email.data,
            'phone': normalized_phone,
            'name': form.name.data,
            'dob': form.dob.data.strftime('%Y-%m-%d'),
            'password': form.password.data
        }

        try:
            verification = twilio_client.verify.v2.services(TWILIO_VERIFY_SID) \
                .verifications.create(to=normalized_phone, channel='sms')
            logging.info(f"Register: OTP sent to {normalized_phone}")
            flash('OTP sent to your phone number.', 'success')
            return redirect(url_for('verify_otp'))
        except Exception as e:
            logging.error(f"Register: Failed to send OTP: {str(e)}")
            flash(f"Failed to send OTP: {str(e)}", 'error')
            return render_template('register.html', form=form)

    return render_template('register.html', form=form)

@app.route("/verify-otp", methods=['GET', 'POST'])
def verify_otp():
    if 'register_data' not in session:
        flash('Session expired. Please register again.', 'error')
        return redirect(url_for('register'))

    form = OTPForm()
    if form.validate_on_submit():
        try:
            phone = session['register_data']['phone']
            verification_check = twilio_client.verify.v2.services(TWILIO_VERIFY_SID) \
                .verification_checks.create(to=phone, code=form.otp.data)

            if verification_check.status == 'approved':
                hashed_password = bcrypt.generate_password_hash(session['register_data']['password']).decode('utf-8')
                user = User(
                    email=session['register_data']['email'],
                    phone=session['register_data']['phone'],
                    name=session['register_data']['name'],
                    dob=session['register_data']['dob'],
                    password_hash=hashed_password
                )
                db.session.add(user)
                db.session.commit()
                session.pop('register_data', None)
                flash('Registration successful! Please log in.', 'success')
                return redirect(url_for('login'))
            else:
                flash('Invalid OTP. Please try again.', 'error')
        except Exception as e:
            logging.error(f"Verify-OTP: OTP verification failed: {str(e)}")
            flash(f"OTP verification failed: {str(e)}", 'error')

    return render_template('verify_otp.html', form=form)

@app.route("/logout")
@login_required
def logout():
    logout_user()
    session.clear()  # Clear all session data
    flash('You have been logged out.', 'success')
    return redirect(url_for('home'))

@app.route("/detect")
@login_required
def detect():
    return render_template('detect.html')

@app.route("/team")
def team():
    return render_template('team.html')

@app.route("/plans")
def plans():
    return render_template('plans.html', plans=PLANS)

@app.route("/payment")
@login_required
def payment():
    plan = request.args.get("plan")
    if plan not in PLANS:
        return redirect(url_for('home'))
    return render_template('payment.html',
                          plan_name=PLANS[plan]["name"],
                          plan_price=PLANS[plan]["price"],
                          plan_id=plan,
                          stripe_publishable_key=os.getenv("STRIPE_PUBLISHABLE_KEY"))

@app.route("/feedback")
@login_required
def feedback():
    try:
        with open(FEEDBACK_FILE, "r") as f:
            feedbacks = json.load(f)
    except:
        feedbacks = []
    return render_template('feedback.html', feedbacks=feedbacks)

@app.route("/history")
@login_required
def history():
    try:
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
        user_history = [item for item in history if item.get('user_id') == current_user.id]
    except:
        user_history = []
    return render_template('history.html', history=user_history)

@app.route("/dashboard")
@login_required
def dashboard():
    try:
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
        user_history = [item for item in history if item.get('user_id') == current_user.id]

        total_detections = len(user_history)
        leaks_detected = sum(1 for item in user_history if item.get('leak', False))

        recent_detections = sorted(
            user_history,
            key=lambda x: x.get('timestamp', ''),
            reverse=True
        )[:5]

        severity_counts = {"Critical": 0, "Moderate": 0, "Low": 0}
        for item in user_history:
            if item.get('leak', False):
                severity = item.get('severity', 'None')
                if severity in severity_counts:
                    severity_counts[severity] += 1

        recent_probs = [item['probability'] for item in user_history[-5:] if item.get('leak')]
        maintenance_risk = 0.0
        if recent_probs and len(recent_probs) > 0:
            maintenance_risk = float(maintenance_model.predict(np.array(recent_probs).reshape(-1, 1)).mean())
            maintenance_risk = max(0.0, min(1.0, maintenance_risk))

        return render_template(
            'dashboard.html',
            total_detections=total_detections,
            leaks_detected=leaks_detected,
            water_saved=water_saved,
            recent_detections=recent_detections,
            severity_counts=severity_counts,
            maintenance_risk=maintenance_risk
        )
    except Exception as e:
        logging.error(f"Dashboard error: {str(e)}")
        flash('Error loading dashboard. Please try again.', 'error')
        return redirect(url_for('home'))

@app.route("/success")
def success():
    session_id = request.args.get("session_id")
    return render_template('success.html', session_id=session_id)

@app.route("/create-checkout-session", methods=["POST"])
@login_required
def create_checkout_session():
    try:
        data = request.get_json()
        plan = data.get("plan")

        if plan not in PLANS or PLANS[plan]["stripe_price_id"] is None:
            return jsonify({"error": "Invalid plan or free plan selected"}), 400

        session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price': PLANS[plan]["stripe_price_id"],
                'quantity': 1,
            }],
            mode='subscription',
            success_url=f"{public_url}/success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{public_url}/payment?plan={plan}",
        )
        return jsonify({"id": session.id})
    except Exception as e:
        logging.error(f"Checkout session error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/model-metrics")
def get_model_metrics():
    if not metrics:
        return jsonify({"error": "Model metrics not available"}), 404
    return jsonify({
        "accuracy": float(metrics.get("accuracy", 0)),
        "precision": float(metrics.get("precision", 0)),
        "recall": float(metrics.get("recall", 0)),
        "last_updated": datetime.now().isoformat()
    })

@app.route("/demo-data")
def get_demo_data():
    global demo_index
    if demo_data is None:
        return jsonify({"error": "Demo dataset not available"}), 404

    row = demo_data.iloc[demo_index].to_dict()
    demo_index = (demo_index + 1) % len(demo_data)
    return jsonify(row)

@app.route("/water-saved")
def get_water_saved():
    return jsonify({"water_saved_liters": water_saved})

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    try:
        data = request.get_json()
        logging.info(f"Predict data received: {data}")

        if not data:
            return jsonify({"error": "No data provided"}), 400

        required_fields = ["Pressure_PSI", "FlowRate_GPM", "Temperature_Cel", "Moisture_Percent", "Acoustic_dB"]
        inputs = {}
        for field in required_fields:
            if field not in data or data[field] is None:
                return jsonify({"error": f"Missing field: {field}"}), 400
            try:
                inputs[field] = float(data[field])
            except ValueError:
                return jsonify({"error": f"Invalid value for {field}"}), 400

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

        severity = "Critical" if probability > 0.75 else "Moderate" if probability > 0.5 else "Low"
        recommendations = {
            "Critical": "Immediate shutdown and emergency repair required",
            "Moderate": "Schedule maintenance within 24 hours",
            "Low": "Monitor and inspect within 48 hours"
        }

        response = {
            "leak": result,
            "probability": probability,
            "severity": severity if result else "None",
            "input_values": inputs,
            "message": f"{severity} leak probability detected!" if result else "No leak detected",
            "model_metrics": metrics,
            "recommendation": recommendations[severity] if result else "No action needed",
            "timestamp": datetime.now().isoformat()
        }

        try:
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
        except:
            history = []

        history.append({
            "user_id": current_user.id,
            "timestamp": response["timestamp"],
            "leak": response["leak"],
            "probability": response["probability"],
            "severity": response["severity"],
            "inputs": response["input_values"],
            "recommendation": response["recommendation"]
        })

        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)

        user_id = current_user.phone
        logging.info(f"Predict: Phone number for Twilio: {user_id}")
        if user_id in alert_timestamps and datetime.now() - alert_timestamps[user_id] < timedelta(minutes=5):
            response["alert"] = "Alert rate limited - please wait 5 minutes"
            return jsonify(response)

        if result:
            flow_rate = inputs["FlowRate_GPM"] * 3.78541  # Convert GPM to LPM
            minutes_saved = 60 if severity == "Critical" else 30 if severity == "Moderate" else 15
            global water_saved
            water_saved += flow_rate * minutes_saved
            response["water_saved"] = water_saved

            with open(WATER_SAVED_FILE, "w") as f:
                json.dump({"water_saved": water_saved}, f)

        if result and not DEBUG_MODE:
            sms_message = (
                f"🚨 {severity} Leak Detected!\n"
                f"Probability: {(probability * 100):.2f}%\n"
                f"Pressure: {inputs['Pressure_PSI']} PSI\n"
                f"Flow Rate: {inputs['FlowRate_GPM']} GPM\n"
                f"Temperature: {inputs['Temperature_Cel']} °C\n"
                f"Moisture: {inputs['Moisture_Percent']} %\n"
                f"Acoustic: {inputs['Acoustic_dB']} dB\n"
                f"Recommended action: {response['recommendation']}"
            )

            try:
                twilio_client.messages.create(
                    body=sms_message,
                    from_=TWILIO_PHONE_NUMBER,
                    to=user_id
                )
                logging.info("SMS sent successfully")
            except Exception as e:
                response["sms_error"] = f"SMS failed: {str(e)}"
                logging.error(f"SMS failed: {str(e)}")

            call_message = (
                f"Alert! A {severity.lower()} leak has been detected with a probability of {probability * 100:.2f} percent. "
                f"Recommended action: {response['recommendation']}. Check your SMS for details."
            )

            try:
                twilio_client.calls.create(
                    to=user_id,
                    from_=TWILIO_PHONE_NUMBER,
                    url=f"{public_url}/twiml?message={urllib.parse.quote(call_message)}"
                )
                logging.info("Call initiated successfully")
            except Exception as e:
                response["call_error"] = f"Call failed: {str(e)}"
                logging.error(f"Call failed: {str(e)}")

            email_body = (
                f"LeakSentry: {severity} Leak Detected\n\n"
                f"Probability: {(probability * 100):.2f}%\n"
                f"Pressure: {inputs['Pressure_PSI']} PSI\n"
                f"Flow Rate: {inputs['FlowRate_GPM']} GPM\n"
                f"Temperature: {inputs['Temperature_Cel']} °C\n"
                f"Moisture: {inputs['Moisture_Percent']} %\n"
                f"Acoustic: {inputs['Acoustic_dB']} dB\n"
                f"Recommended action: {response['recommendation']}\n\n"
                f"View your detection history at {public_url}/history"
            )

            try:
                if send_email(current_user.email, f"LeakSentry: {severity} Leak Alert", email_body):
                    response["email_status"] = "Email sent successfully"
                else:
                    response["email_error"] = "Failed to send email"
            except Exception as e:
                response["email_error"] = f"Email failed: {str(e)}"
                logging.error(f"Email failed: {str(e)}")

            if severity == "Critical":
                escalation_message = (
                    f"CRITICAL LEAK ALERT: User {current_user.email} (Phone: {user_id})\n"
                    f"Probability: {(probability * 100):.2f}%\n"
                    f"Details: {json.dumps(inputs)}\n"
                    f"Action: Immediate intervention required."
                )
                try:
                    twilio_client.messages.create(
                        body=escalation_message,
                        from_=TWILIO_PHONE_NUMBER,
                        to=ADMIN_PHONE
                    )
                    logging.info("Admin SMS sent successfully")
                except Exception as e:
                    logging.error(f"Admin SMS failed: {str(e)}")

                try:
                    send_email(ADMIN_EMAIL, "CRITICAL LEAK ALERT", escalation_message)
                    logging.info("Admin email sent successfully")
                except Exception as e:
                    logging.error(f"Admin email failed: {str(e)}")

            alert_timestamps[user_id] = datetime.now()
        elif DEBUG_MODE and result:
            logging.info("Debug mode: Alerts would be sent:")
            logging.info(f"SMS: {sms_message}")
            logging.info(f"Call message: {call_message}")
            logging.info(f"Email: {email_body}")
            if severity == "Critical":
                logging.info(f"Escalation: {escalation_message}")

        return jsonify(response)
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

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
        return str(VoiceResponse().say(f"Error: {str(e)}")), 500

@app.route("/predict/live", methods=["POST"])
@login_required
def predict_live():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        required_fields = ["Pressure_PSI", "FlowRate_GPM", "Temperature_Cel", "Moisture_Percent", "Acoustic_dB"]
        inputs = {k: float(data.get(k, 0)) for k in required_fields}

        input_features = np.array([[inputs[k] for k in required_fields]])
        probability = float(model.predict_proba(input_features)[0][1])

        return jsonify({
            "probability": probability,
            "severity": "High" if probability > 0.75 else "Medium" if probability > 0.5 else "Low"
        })
    except Exception as e:
        logging.error(f"Live prediction error: {str(e)}")
        return jsonify({"error": f"Live prediction error: {str(e)}"}), 500

@app.route("/feature-importance")
def feature_importance():
    try:
        if hasattr(model, 'feature_importances_'):
            features = ["Pressure", "Flow Rate", "Temperature", "Moisture", "Acoustic"]
            importance = model.feature_importances_.tolist()
            return jsonify({"features": features, "importance": importance})
        return jsonify({"error": "Feature importance not available"}), 404
    except Exception as e:
        logging.error(f"Feature importance error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/submit-feedback", methods=["POST"])
@login_required
def submit_feedback():
    try:
        name = request.form.get("name")
        email = request.form.get("email")
        feedback = request.form.get("feedback")

        if not all([name, email, feedback]):
            return jsonify({"error": "All fields are required"}), 400

        try:
            with open(FEEDBACK_FILE, "r") as f:
                feedbacks = json.load(f)
        except:
            feedbacks = []

        feedbacks.append({
            "user_id": current_user.id,
            "name": name,
            "email": email,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat()
        })

        with open(FEEDBACK_FILE, "w") as f:
            json.dump(feedbacks, f, indent=2)

        logging.info(f"Feedback received from {name} ({email}): {feedback}")
        return jsonify({"message": "Thank you for your feedback!"}), 200
    except Exception as e:
        logging.error(f"Feedback submission error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        logging.info(f"Chat endpoint received: {data}")
        if not data or "message" not in data:
            logging.error("No message provided in request")
            return jsonify({"error": "No message provided"}), 400

        user_message = data["message"].lower().strip()
        logging.info(f"Processing chatbot message: {user_message}")

        response = CHATBOT_RESPONSES.get("default")
        for key, value in CHATBOT_RESPONSES.items():
            if key == "default":
                continue
            keywords = key.split("|")
            if any(keyword in user_message for keyword in keywords):
                response = value
                break

        logging.info(f"Chatbot response: {response}")
        return jsonify({"response": response})
    except Exception as e:
        logging.error(f"Chatbot error: {str(e)}")
        return jsonify({"error": f"Chatbot error: {str(e)}"}), 500

# Function to generate QR code at startup
def generate_qr_code_on_startup():
    qr_url = f"{public_url}/register"
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(qr_url)
    qr.make(fit=True)
    img = qr.make_image(fill='black', back_color='white')
    qr_path = os.path.join(app.static_folder, 'images', 'qr_code.png')
    os.makedirs(os.path.dirname(qr_path), exist_ok=True)
    img.save(qr_path)
    logging.info(f"QR code generated and saved to {qr_path} with URL: {qr_url}")

# Route for mobile access
@app.route('/mobile')
def mobile():
    qr_url = f"{public_url}/register"
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(qr_url)
    qr.make(fit=True)
    img = qr.make_image(fill='black', back_color='white')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return send_file(
        img_byte_arr,
        mimetype='image/png',
        as_attachment=True,
        download_name='qr_code.png',
        cache_timeout=86400
    )

# Global variable to store the public URL
public_url = None

if __name__ == "__main__":
    # Attempt to terminate all ngrok processes
    try:
        ngrok.kill()
        logging.info("Terminated existing pyngrok tunnels.")
    except Exception as e:
        logging.warning(f"Failed to kill pyngrok tunnels: {e}")

    # Try to kill standalone ngrok processes (Windows-specific)
    try:
        subprocess.run(["taskkill", "/F", "/IM", "ngrok.exe"], capture_output=True, text=True)
        logging.info("Terminated any standalone ngrok processes.")
    except Exception as e:
        logging.warning(f"Failed to terminate standalone ngrok processes: {e}")

    # Start ngrok tunnel
    try:
        ngrok_tunnel = ngrok.connect(5000, bind_tls=True)
        public_url = ngrok_tunnel.public_url
        logging.info(f"ngrok tunnel started: {public_url}")
    except Exception as e:
        logging.error(f"Failed to start ngrok tunnel: {e}")
        public_url = "http://localhost:5000"
        logging.warning(f"Fallback to local URL: {public_url}")

    print(f"Using public URL: {public_url}")

    # Generate QR code on server startup
    generate_qr_code_on_startup()

    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)