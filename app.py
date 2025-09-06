from flask import Flask, render_template, request, redirect, url_for, session, flash, abort
import joblib
import pandas as pd
import os
from datetime import datetime
from pyproj import Proj, Transformer
import folium
from folium.plugins import HeatMap
from functools import wraps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io 
import base64
import numpy as np
# import json # currently not in use
import sqlite3   # ✅ Added
import ast   # ✅ add at the top with other imports

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'forest_fire_prediction_secret_key_2024'  # Important for session management

# Load trained ML model from the classification task
model = joblib.load("forest_fire_model.pkl")

# ✅ SQLite database file
DB_FILE = "users.db"

# List of protected routes that require authentication
PROTECTED_ROUTES = ['home', 'try_page', 'history_page', 'stats_btn', 'real_time_stats', 
                   'map_view', 'profile', 'logout']

# Add stats_btn to restricted direct access routes
RESTRICTED_DIRECT_ACCESS = ['stats_btn']

# ✅ Create tables if not exists
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL
                )''')
    # ✅ Prediction history table
    c.execute('''CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    inputs TEXT NOT NULL,
                    result TEXT NOT NULL,
                    probability REAL NOT NULL
                )''')
    conn.commit()
    conn.close()

# ✅ Database helpers
def add_user(username, password, name, email):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO users (username, password, name, email) VALUES (?, ?, ?, ?)",
              (username, password, name, email))
    conn.commit()
    conn.close()

def get_user(username, password):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    user = c.fetchone()
    conn.close()
    return user

def get_user_by_username(username):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=?", (username,))
    user = c.fetchone()
    conn.close()
    return user

def user_exists(username, email):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? OR email=?", (username, email))
    user = c.fetchone()
    conn.close()
    return user is not None

# ✅ Save prediction history to SQLite
def save_to_history(input_data, result, probability):
    history_entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'inputs': input_data,
        'result': result,
        'probability': probability,
        'user': session.get('username', 'guest')
    }
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO history (username, timestamp, inputs, result, probability) VALUES (?, ?, ?, ?, ?)",
              (history_entry['user'], history_entry['timestamp'], str(input_data), result, probability))
    conn.commit()
    conn.close()
    return history_entry

# ✅ Get all history
def get_history():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT username, timestamp, inputs, result, probability FROM history ORDER BY id DESC LIMIT 50")
    rows = c.fetchall()
    conn.close()
    history = []
    for row in rows:
        history.append({
            'user': row[0],
            'timestamp': row[1],
            'inputs': row[2],
            'result': row[3],
            'probability': row[4]
        })
    return history

# ✅ Get history for a specific user
def get_user_history(username):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT timestamp, inputs, result, probability FROM history WHERE username=? ORDER BY id DESC LIMIT 50", (username,))
    rows = c.fetchall()
    conn.close()
    history = []
    for row in rows:
        try:
            inputs_dict = ast.literal_eval(row[1])   # ✅ safely convert string back to dict
            print("DEBUG INPUTS:", type(inputs_dict), inputs_dict)  # ✅ Added debug print
        except Exception as e:
            print(f"DEBUG ERROR: Could not parse inputs - {e}")
            inputs_dict = {}
        history.append({
            'timestamp': row[0],
            'inputs': inputs_dict,
            'result': row[2],
            'probability': row[3]
        })
    return history


# Require login for certain routes
def login_required(f):
    """Decorator to require login for specific routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            flash('Please login to access this page', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.before_request
def check_authentication():
    """Check authentication before every request"""
    if request.endpoint and (
        request.endpoint.startswith('static') or 
        request.endpoint in ['login', 'register', 'index'] or
        'auth' in request.endpoint
    ):
        return
    if request.endpoint in PROTECTED_ROUTES and 'username' not in session:
        flash('Please login to access this page', 'error')
        return redirect(url_for('login'))
    if request.endpoint in RESTRICTED_DIRECT_ACCESS and request.referrer is None:
        abort(403)

def generate_boxplot(column_name, label=None):
    df = pd.read_csv("forestfires_augmented.csv")  # ✅ using augmented dataset
    data = df[column_name]
    Q1 = data.quantile(0.25)
    Q2 = data.median()
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    min_val = data.min()
    max_val = data.max()
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    plt.figure(figsize=(6, 6))
    ax = sns.boxplot(y=data, color="skyblue", fliersize=5)
    plt.title(f"Box Plot of {label or column_name}", fontsize=14)
    plt.ylabel(label or column_name)
    for val in [Q1, Q2, Q3]:
        ax.axhline(val, color="gray", linestyle="--", linewidth=0.8)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return {
        "Q1": round(Q1, 2),
        "Q2": round(Q2, 2),
        "Q3": round(Q3, 2),
        "IQR": round(IQR, 2),
        "Min": round(min_val, 2),
        "Max": round(max_val, 2),
        "Outliers": outliers.tolist(),
        "Plot": img_base64
    }

# ------------------ AUTH ROUTES ------------------
@app.route("/")
def index():
    return redirect(url_for('login'))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        # Clear old session only when new login attempt is made
        session.clear()

        user = get_user(username, password)
        if user:
            session["username"] = username
            flash(f"Welcome {username}!", "success")
            return redirect(url_for("home"))
        flash("Invalid username or password", "danger")

    return render_template("login.html")



@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        # Clear only when user submits form
        session.clear()

        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        name = request.form.get('name')
        email = request.form.get('email')

        if not username or not password or not confirm_password or not name or not email:
            flash('All fields are required', 'error')
            return render_template("register.html")
        if user_exists(username, email):
            flash('Username or Email already exists', 'error')
            return render_template("register.html")
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template("register.html")

        add_user(username, password, name, email)
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))

    return render_template("register.html")



@app.route("/logout")
@login_required
def logout():
    session.pop('username', None)
    session.pop('name', None)
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

@app.route("/home")
@login_required
def home():
    result = request.args.get('result')
    return render_template("index.html", result=result, username=session.get('username'))

# ------------------ PREDICTION & HISTORY ------------------
@app.route("/try", methods=["GET", "POST"])
@login_required
def try_page():
    result = None
    probability = None
    temperature = None
    humidity = None
    rainfall = None
    wind_speed = None
    ffmc = None
    dmc = None
    dc = None
    isi = None
    if request.method == "POST":
        try:
            temperature = float(request.form.get("temperature"))
            humidity = float(request.form.get("humidity"))
            rainfall = float(request.form.get("rainfall"))
            wind_speed = float(request.form.get("wind_speed"))
            ffmc = float(request.form.get("ffmc"))
            dmc = float(request.form.get("dmc"))
            dc = float(request.form.get("dc"))
            isi = float(request.form.get("isi"))
            input_data = pd.DataFrame([[temperature, humidity, wind_speed, rainfall, ffmc, dmc, dc, isi]],
                                      columns=["temp", "RH", "wind", "rain", "FFMC", "DMC", "DC", "ISI"])
            fire_probability = model.predict_proba(input_data)[0][1]
            if fire_probability >= 0.5:  
                risk_level = "HIGH"
            elif fire_probability >= 0.2:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            input_data_dict = {
                'temperature': temperature,
                'humidity': humidity,
                'rainfall': rainfall,
                'wind_speed': wind_speed,
                'ffmc': ffmc,
                'dmc': dmc,
                'dc': dc,
                'isi': isi
            }
            save_to_history(input_data_dict, risk_level, float(fire_probability))
            result = risk_level
            probability = fire_probability
            flash(f'Prediction completed: {risk_level} risk detected', 'success')
        except Exception as e:
            flash(f'An error occurred: {e}. Please check your inputs.', 'error')
    return render_template("try.html", 
                           result=result, 
                           probability=probability, 
                           temperature=temperature,
                           humidity=humidity,
                           rainfall=rainfall,
                           wind_speed=wind_speed,
                           ffmc=ffmc,
                           dmc=dmc,
                           dc=dc,
                           isi=isi,
                           username=session.get('username'))

@app.route("/history")
@login_required
def history_page():
    username = session.get('username')
    history = get_user_history(username)
    return render_template("history.html", history=history, username=username)

@app.route("/stats_btn")
@login_required
def stats_btn():
    if not request.referrer or 'try' not in request.referrer:
        abort(403)
    return render_template('stats_btn.html', 
                           temperature=request.args.get('temperature'),
                           humidity=request.args.get('humidity'),
                           rainfall=request.args.get('rainfall'),
                           wind_speed=request.args.get('wind_speed'),
                           ffmc=request.args.get('ffmc'),
                           dmc=request.args.get('dmc'),
                           dc=request.args.get('dc'),
                           isi=request.args.get('isi'),
                           username=session.get('username'))

@app.route("/real")
@login_required
def real_time_stats():
    return render_template('real.html', username=session.get('username'))

@app.route('/map')
@login_required
def map_view():
    df = pd.read_csv("forestfires.csv")  # ⚠️ still original dataset for map
    if 'temp' not in df.columns:
        return "Error: 'temp' column not found in dataset."
    df['lat'] = 41.8 + (df['Y'] - df['Y'].mean()) * 0.01
    df['lon'] = -6.7 + (df['X'] - df['X'].mean()) * 0.01
    heat_data = [[row['lat'], row['lon'], row['temp']] for _, row in df.iterrows()]
    m = folium.Map(location=[41.8, -6.7], zoom_start=12)
    HeatMap(heat_data, radius=15, blur=10, min_opacity=0.5).add_to(m)
    map_html = m._repr_html_()
    return render_template("map.html", map_html=map_html, username=session.get('username'))

@app.route('/profile')
@login_required
def profile():
    username = session.get('username')
    user = get_user_by_username(username)
    return render_template('profile.html', 
                           username=username, 
                           name=user[3] if user else "",
                           email=user[4] if user else "",
                           prediction_count=len(get_user_history(username)))

# ------------------ ERROR HANDLERS ------------------
@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404

@app.errorhandler(403)
def forbidden(e):
    return render_template("403.html"), 403

@app.errorhandler(500)
def server_error(e):
    return render_template("500.html"), 500

# ------------------ EXTRA ROUTES ------------------
@app.route('/options')
def options():
    return render_template("options.html")

@app.route('/temp')
def temp_boxplot():
    result = generate_boxplot("temp", "Temperature")
    return render_template("temp.html", **result)

@app.route('/rh')
def rh_boxplot():
    result = generate_boxplot("RH", "Relative Humidity")
    return render_template("rh.html", **result)

@app.route('/wind')
def wind_boxplot():
    result = generate_boxplot("wind", "Wind Speed")
    return render_template("wind.html", **result)

@app.route("/rain")
def rain_boxplot():
    result = generate_boxplot("rain", "Rain")
    return render_template("rain.html", **result)

@app.route("/ffmc")
def ffmc_boxplot():
    result = generate_boxplot("FFMC", "FFMC")
    return render_template("ffmc.html", **result)

@app.route("/dmc")
def dmc_boxplot():
    result = generate_boxplot("DMC", "DMC")
    return render_template("dmc.html", **result)

@app.route("/dc")
def dc_boxplot():
    result = generate_boxplot("DC", "DC")
    return render_template("dc.html", **result)

@app.route("/isi")
def isi_boxplot():
    result = generate_boxplot("ISI", "ISI")
    return render_template("isi.html", **result)

@app.route("/monthly_risk")
@login_required
def monthly_risk():
    df = pd.read_csv("forestfires.csv")
    month_map = {
        "jan": "January", "feb": "February", "mar": "March", "apr": "April",
        "may": "May", "jun": "June", "jul": "July", "aug": "August",
        "sep": "September", "oct": "October", "nov": "November", "dec": "December"
    }
    df["Month"] = df["month"].str.lower().map(month_map)
    def classify_risk(area):
        if area == 0:
            return "Low"
        elif area <= 10:
            return "Medium"
        else:
            return "High"
    df["Risk"] = df["area"].apply(classify_risk)
    monthly_risk = df.groupby("Month")["Risk"].value_counts().unstack(fill_value=0).reset_index()
    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    monthly_risk["Month"] = pd.Categorical(monthly_risk["Month"], categories=month_order, ordered=True)
    monthly_risk = monthly_risk.sort_values("Month")
    monthly_risk = monthly_risk.rename(columns={
        "Low": "Low Risk",
        "Medium": "Medium Risk",
        "High": "High Risk"
    })
    table_data = monthly_risk.to_dict(orient="records")
    return render_template(
        "monthly_risk.html",
        table_data=table_data,
        username=session.get('username')
    )

# ✅ New route for test.html
@app.route("/test")
def test():
    return render_template("test.html")

@app.route('/features_info')
def features_info():
    return render_template("features_info.html")

# ------------------ MAIN ------------------
if __name__ == '__main__':
    init_db()   # ✅ Ensure DB is created
    app.run(debug=True, use_reloader=True)
