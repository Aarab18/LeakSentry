from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, TelField, DateField
from wtforms.validators import DataRequired, Email, Length, EqualTo

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

class RegisterForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    phone = TelField('Phone Number', validators=[DataRequired()])
    name = StringField('Full Name', validators=[DataRequired()])
    dob = DateField('Date of Birth', validators=[DataRequired()], format='%Y-%m-%d')
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Send OTP')

class OTPForm(FlaskForm):
    otp = StringField('OTP', validators=[DataRequired(), Length(min=4, max=6)])
    submit = SubmitField('Verify OTP')