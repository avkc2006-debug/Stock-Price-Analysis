"""
StockSage - Stock Price Prediction Web App
Run with:  python app.py
Then open: http://127.0.0.1:5000
"""

import os
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")  # headless backend, must be set before pyplot import
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
import joblib

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user,
    login_required, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "stock_price_model.pkl")
CHART_PATH = os.path.join(BASE_DIR, "static", "images", "chart.png")

# ---------------------------------------------------------------------------
# App & config
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = "stocksage-dev-secret-change-me"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(BASE_DIR, "stocksage.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message = "Please log in to access that page."
login_manager.login_message_category = "warning"

app.jinja_env.globals.update(zip=zip)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    ticker = db.Column(db.String(20), nullable=False)
    days = db.Column(db.Integer, nullable=False)
    last_price = db.Column(db.Float, nullable=False)
    predicted_price = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


with app.app_context():
    db.create_all()


# ---------------------------------------------------------------------------
# Prediction engine
# ---------------------------------------------------------------------------
def train_and_predict(ticker: str, days_ahead: int):
    """
    Downloads ~2 years of history for `ticker`, fits a simple linear
    regression model on it, forecasts `days_ahead` days into the future,
    saves a chart, and also saves the freshly trained model to
    stock_price_model.pkl (so the project always ships with a real,
    up-to-date model file matching your folder structure).
    """
    end = datetime.today()
    start = end - timedelta(days=365 * 2)

    data = yf.download(ticker, start=start, end=end, progress=False)
    if data.empty:
        return None

    data = data.reset_index()
    data["DateOrdinal"] = data["Date"].map(lambda d: d.toordinal())

    X = data[["DateOrdinal"]].values
    y = data["Close"].values.ravel()

    model = LinearRegression()
    model.fit(X, y)

    # Persist the model so stock_price_model.pkl always reflects the
    # most recently requested ticker's trained model.
    joblib.dump(model, MODEL_PATH)

    last_date = data["Date"].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_ahead + 1)]
    future_ordinals = np.array([[d.toordinal()] for d in future_dates])
    future_preds = model.predict(future_ordinals)

    # Chart
    plt.figure(figsize=(10, 5))
    plt.plot(data["Date"], y, label="Historical Close", color="#2563eb")
    plt.plot(future_dates, future_preds, label="Predicted", linestyle="--", color="#f97316")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.title(f"{ticker.upper()} — Price Forecast")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(CHART_PATH)
    plt.close()

    return {
        "ticker": ticker.upper(),
        "last_price": round(float(y[-1]), 2),
        "predicted_price": round(float(future_preds[-1]), 2),
        "days": days_ahead,
        "dates": [d.strftime("%Y-%m-%d") for d in future_dates],
        "prices": [round(float(p), 2) for p in future_preds],
        "chart_url": url_for("static", filename="images/chart.png"),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/services")
def services():
    return render_template("services.html")


@app.route("/faq")
def faq():
    return render_template("faq.html")


@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        flash("Thanks for reaching out! We'll get back to you soon.", "success")
        return redirect(url_for("contact"))
    return render_template("contact.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if not username or not email or not password:
            flash("All fields are required.", "danger")
        elif User.query.filter_by(username=username).first():
            flash("That username is already taken.", "danger")
        elif User.query.filter_by(email=email).first():
            flash("That email is already registered.", "danger")
        else:
            user = User(username=username, email=email)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            flash("Account created! Please log in.", "success")
            return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user)
            flash("Welcome back!", "success")
            next_page = request.args.get("next")
            return redirect(next_page or url_for("dashboard"))
        flash("Invalid username or password.", "danger")

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))


@app.route("/dashboard")
@login_required
def dashboard():
    history = (
        Prediction.query.filter_by(user_id=current_user.id)
        .order_by(Prediction.created_at.desc())
        .limit(20)
        .all()
    )
    return render_template("dashboard.html", history=history)


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    if request.method == "POST":
        ticker = request.form.get("ticker", "").strip().upper()
        days = request.form.get("days", "7")

        try:
            days = int(days)
            days = max(1, min(days, 60))
        except ValueError:
            days = 7

        if not ticker:
            flash("Please enter a stock ticker symbol (e.g. AAPL, TSLA, INFY.NS).", "danger")
            return redirect(url_for("predict"))

        result = train_and_predict(ticker, days)

        if result is None:
            flash(f"Couldn't find data for '{ticker}'. Check the symbol and try again.", "danger")
            return redirect(url_for("predict"))

        record = Prediction(
            user_id=current_user.id,
            ticker=result["ticker"],
            days=result["days"],
            last_price=result["last_price"],
            predicted_price=result["predicted_price"],
        )
        db.session.add(record)
        db.session.commit()

        return render_template("result.html", result=result)

    return render_template("predict.html")


@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", code=404, message="Page not found."), 404


@app.errorhandler(500)
def server_error(e):
    return render_template("error.html", code=500, message="Something went wrong on our end."), 500


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
