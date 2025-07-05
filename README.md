LeakSentry
Welcome to LeakSentry, an AI-powered web application designed to detect and manage water leaks with precision using machine learning. This project leverages advanced models to analyze pressure, flow rate, temperature, moisture, and acoustic data, providing real-time predictions and alerts via Twilio. Built with Flask, it offers a user-friendly interface with subscription plans for individuals and enterprises.
Features

Leak Detection: AI-driven analysis of sensor data to predict leak probabilities.
Real-Time Updates: Live probability display and interactive charts.
Alerts: SMS notifications via Twilio for detected leaks.
Subscription Plans: Basic, Pro, and Enterprise tiers with Stripe integration.
User Interface: Responsive design with dark/light theme toggle.
Demo Mode: Simulated data for testing and demonstration.

Installation
Prerequisites

Python 3.8+
Git
Node.js (for frontend development, if needed)

Setup

Clone the Repository
git clone https://github.com/yourusername/leaksentry.git
cd leaksentry


Install DependenciesCreate a virtual environment and install required packages:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt


Configure Environment VariablesCreate a .env file in the root directory and add the following:
SECRET_KEY=your-secret-key
TWILIO_ACCOUNT_SID=your-twilio-sid
TWILIO_AUTH_TOKEN=your-twilio-token
TWILIO_PHONE_NUMBER=+1234567890
USER_PHONE_NUMBER=+0987654321
STRIPE_SECRET_KEY=sk_test_xxx
STRIPE_PUBLISHABLE_KEY=pk_test_xxx

Replace the values with your actual credentials.

Prepare the ModelEnsure the leak_detection_model.pkl file is placed in the models/ directory (train your model or use the provided one).

Run the ApplicationStart the Flask server:
python app.py

Access the app at http://localhost:5000.


Usage

Home Page: View an overview and subscription plans (/).
Detective Page: Input sensor data and detect leaks (/detective).
Payment Page: Subscribe to a plan (/payment?plan={basic|pro|enterprise}).


Contributing

Fork the repository.
Create a feature branch (git checkout -b feature-name).
Commit changes (git commit -m "Add feature-name").
Push to the branch (git push origin feature-name).
Open a Pull Request.


Contact

Email: ahmadaarab315@gmail.com

Acknowledgments

Built by Aarab, Kushagra, and Aayush.
Powered by xAI and open-source technologies.
