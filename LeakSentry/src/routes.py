from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_cors import CORS
from src.config import Config
from src.models import db, User
from src.forms import LoginForm, RegisterForm, OTPForm
from src.utils import send_email, twilio_client, create_verify_service, TWILIO_VERIFY_SID, validate_phone_number, alert_timestamps, demo_data, demo_index, water_saved, MODEL_PATH, metrics, PLANS, CHATBOT_RESPONSES, FEEDBACK_FILE, HISTORY_FILE, DEBUG_MODE, twiml_response, PUBLIC_URL, TWILIO_PHONE_NUMBER, WATER_SAVED_FILE, model
from flask_bcrypt import Bcrypt
import os
import logging
import stripe
import datetime
import json
import numpy as np
import urllib.parse
from datetime import timedelta

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config.from_object(Config)
CORS(app)
db.init_app(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Ensure TWILIO_VERIFY_SID is set
TWILIO_VERIFY_SID = create_verify_service()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

with app.app_context():
    db.create_all()

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password_hash, form.password.data):
            login_user(user)
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
        
        is_valid, result = validate_phone_number(form.phone.data)
        if not is_valid:
            flash(result, 'error')
            return render_template('register.html', form=form)
        
        session.clear()
        session['register_data'] = {
            'email': form.email.data,
            'phone': result,
            'name': form.name.data,
            'dob': form.dob.data.strftime('%Y-%m-%d'),
            'password': form.password.data
        }
        
        try:
            verification = twilio_client.verify.v2.services(TWILIO_VERIFY_SID).verifications.create(to=result, channel='sms')
            flash('OTP sent to your phone number.', 'success')
            return redirect(url_for('verify_otp'))
        except Exception as e:
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
            verification_check = twilio_client.verify.v2.services(TWILIO_VERIFY_SID).verification_checks.create(to=phone, code=form.otp.data)
            if verification_check.status == 'approved':
                hashed_password = bcrypt.generate_password_hash(session['register_data']['password']).decode('utf-8')
                user = User(email=session['register_data']['email'], phone=session['register_data']['phone'],
                            name=session['register_data']['name'], dob=session['register_data']['dob'],
                            password_hash=hashed_password)
                db.session.add(user)
                db.session.commit()
                session.pop('register_data', None)
                flash('Registration successful! Please log in.', 'success')
                return redirect(url_for('login'))
            else:
                flash('Invalid OTP. Please try again.', 'error')
        except Exception as e:
            flash(f"OTP verification failed: {str(e)}", 'error')
    return render_template('verify_otp.html', form=form)

@app.route("/logout")
@login_required
def logout():
    logout_user()
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
    return render_template('payment.html', plan_name=PLANS[plan]["name"], plan_price=PLANS[plan]["price"],
                          plan_id=plan, stripe_publishable_key=os.getenv("STRIPE_PUBLISHABLE_KEY"))

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
        session = stripe.checkout.Session.create(payment_method_types=['card'], line_items=[{
            'price': PLANS[plan]["stripe_price_id"], 'quantity': 1}], mode='subscription',
            success_url=PUBLIC_URL + '/success?session_id={CHECKOUT_SESSION_ID}',
            cancel_url=PUBLIC_URL + '/payment?plan=' + plan)
        return jsonify({"id": session.id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/model-metrics")
def get_model_metrics():
    if not metrics:
        return jsonify({"error": "Model metrics not available"}), 404
    return jsonify({"accuracy": float(metrics.get("accuracy", 0)), "precision": float(metrics.get("precision", 0)),
                    "recall": float(metrics.get("recall", 0)), "last_updated": datetime.datetime.now().isoformat()})

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
    global water_saved
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        required_fields = ["Pressure_PSI", "FlowRate_GPM", "Temperature_Cel", "Moisture_Percent", "Acoustic_dB"]
        inputs = {field: float(data[field]) for field in required_fields if field in data and data[field] is not None}
        if len(inputs) != len(required_fields):
            missing = [field for field in required_fields if field not in data or data[field] is None]
            return jsonify({"error": f"Missing field: {', '.join(missing)}"}), 400
        validation_rules = {"Pressure_PSI": (0, 1000, "Pressure must be between 0 and 1000 PSI"),
                            "FlowRate_GPM": (0, 100, "Flow Rate must be between 0 and 100 GPM"),
                            "Temperature_Cel": (-50, 100, "Temperature must be between -50 and 100 °C"),
                            "Moisture_Percent": (0, 100, "Moisture must be between 0 and 100 %"),
                            "Acoustic_dB": (0, 200, "Acoustic level must be between 0 and 200 dB")}
        for field, (min_val, max_val, error_msg) in validation_rules.items():
            if not (min_val <= inputs[field] <= max_val):
                return jsonify({"error": error_msg}), 400
        input_features = np.array([[inputs[k] for k in required_fields]])
        prediction = model.predict(input_features)
        probability = float(model.predict_proba(input_features)[0][1])
        result = bool(prediction[0])
        severity = "Critical" if probability > 0.75 else "Moderate" if probability > 0.5 else "Low"
        recommendations = {"Critical": "Immediate shutdown and emergency repair required",
                          "Moderate": "Schedule maintenance within 24 hours",
                          "Low": "Monitor and inspect within 48 hours"}
        response = {"leak": result, "probability": probability, "severity": severity if result else "None",
                    "input_values": inputs, "message": f"{severity} leak probability detected!" if result else "No leak detected",
                    "model_metrics": metrics, "recommendation": recommendations[severity] if result else "No action needed",
                    "timestamp": datetime.datetime.now().isoformat()}
        try:
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
        except:
            history = []
        history.append({"user_id": current_user.id, "timestamp": response["timestamp"], "leak": response["leak"],
                        "probability": response["probability"], "severity": response["severity"], "inputs": response["input_values"],
                        "recommendation": response["recommendation"]})
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
        user_id = current_user.phone
        if user_id in alert_timestamps and datetime.datetime.now() - alert_timestamps[user_id] < timedelta(minutes=5):
            response["alert"] = "Alert rate limited - please wait 5 minutes"
            return jsonify(response)
        if result:
            flow_rate = inputs["FlowRate_GPM"] * 3.78541
            minutes_saved = 60 if severity == "Critical" else 30 if severity == "Moderate" else 15
            global water_saved
            water_saved += flow_rate * minutes_saved
            response["water_saved"] = water_saved
            with open(WATER_SAVED_FILE, "w") as f:
                json.dump({"water_saved": water_saved}, f)
            sms_message = (f"🚨 {severity} Leak Detected!\nProbability: {(probability * 100):.2f}%\nPressure: {inputs['Pressure_PSI']} PSI\n"
                          f"Flow Rate: {inputs['FlowRate_GPM']} GPM\nTemperature: {inputs['Temperature_Cel']} °C\n"
                          f"Moisture: {inputs['Moisture_Percent']} %\nAcoustic: {inputs['Acoustic_dB']} dB\n"
                          f"Recommended action: {response['recommendation']}")
            try:
                twilio_client.messages.create(body=sms_message, from_=TWILIO_PHONE_NUMBER, to=user_id)
            except Exception as e:
                response["sms_error"] = f"SMS failed: {str(e)}"
            call_message = (f"Alert! A {severity.lower()} leak has been detected with a probability of {probability * 100:.2f} percent. "
                           f"Recommended action: {response['recommendation']}. Check your SMS for details.")
            try:
                twilio_client.calls.create(to=user_id, from_=TWILIO_PHONE_NUMBER,
                                          url=f"{PUBLIC_URL}/twiml?message={urllib.parse.quote(call_message)}")
            except Exception as e:
                response["call_error"] = f"Call failed: {str(e)}"
            try:
                if send_email(current_user.email, f"LeakSentry: {severity} Leak Alert",
                             f"LeakSentry: {severity} Leak Detected\n\nProbability: {(probability * 100):.2f}%\n"
                             f"Pressure: {inputs['Pressure_PSI']} PSI\nFlow Rate: {inputs['FlowRate_GPM']} GPM\n"
                             f"Temperature: {inputs['Temperature_Cel']} °C\nMoisture: {inputs['Moisture_Percent']} %\n"
                             f"Acoustic: {inputs['Acoustic_dB']} dB\nRecommended action: {response['recommendation']}\n\n"
                             f"View your detection history at {PUBLIC_URL}/history"):
                    response["email_status"] = "Email sent successfully"
                else:
                    response["email_error"] = "Failed to send email"
            except Exception as e:
                response["email_error"] = f"Email failed: {str(e)}"
            alert_timestamps[user_id] = datetime.datetime.now()
        elif DEBUG_MODE and result:
            logging.info("Debug mode: Alerts would be sent:")
            logging.info(f"SMS: {sms_message}")
            logging.info(f"Call message: {call_message}")
        return jsonify(response)
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

@app.route("/twiml")
def twiml():
    message = request.args.get('message', 'A leak has been detected.')
    return twiml_response(message)

@app.route("/predict/live", methods=["POST"])
@login_required
def predict_live():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        required_fields = ["Pressure_PSI", "FlowRate_GPM", "Temperature_Cel", "Moisture_Percent", "Acoustic_dB"]
        inputs = {field: float(data.get(field, 0)) for field in required_fields}
        input_features = np.array([[inputs[k] for k in required_fields]])
        probability = float(model.predict_proba(input_features)[0][1])
        return jsonify({"probability": probability, "severity": "High" if probability > 0.75 else "Medium" if probability > 0.5 else "Low"})
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
        recent_detections = sorted(user_history, key=lambda x: x.get('timestamp', ''), reverse=True)[:5]
        severity_counts = {"Critical": 0, "Moderate": 0, "Low": 0}
        for item in user_history:
            if item.get('leak', False):
                severity = item.get('severity', 'None')
                if severity in severity_counts:
                    severity_counts[severity] += 1
        return render_template('dashboard.html', total_detections=total_detections, leaks_detected=leaks_detected,
                              water_saved=water_saved, recent_detections=recent_detections, severity_counts=severity_counts)
    except Exception as e:
        logging.error(f"Dashboard error: {str(e)}")
        flash('Error loading dashboard. Please try again.', 'error')
        return redirect(url_for('home'))

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
        feedbacks.append({"user_id": current_user.id, "name": name, "email": email, "feedback": feedback,
                         "timestamp": datetime.datetime.now().isoformat()})
        with open(FEEDBACK_FILE, "w") as f:
            json.dump(feedbacks, f, indent=2)
        return jsonify({"message": "Thank you for your feedback!"}), 200
    except Exception as e:
        logging.error(f"Feedback submission error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"error": "No message provided"}), 400
        user_message = data["message"].lower().strip()
        response = CHATBOT_RESPONSES.get("default")
        for key, value in CHATBOT_RESPONSES.items():
            if key == "default":
                continue
            keywords = key.split("|")
            if any(keyword in user_message for keyword in keywords):
                response = value
                break
        return jsonify({"response": response})
    except Exception as e:
        logging.error(f"Chatbot error: {str(e)}")
        return jsonify({"error": f"Chatbot error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)