from flask import Flask, render_template, request, redirect, send_file
import joblib
import csv
from collections import Counter
from datetime import datetime
from reportlab.pdfgen import canvas
import os

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")


# =========================
# LOGIN PAGE
# =========================

@app.route("/")
def login_page():
    return render_template("login.html")


@app.route("/login", methods=["POST"])
def login():

    username = request.form.get("username")
    password = request.form.get("password")

    if username == "admin" and password == "jagan2007":
        return redirect("/dashboard")

    return "Invalid username or password"


# =========================
# DASHBOARD
# =========================

@app.route("/dashboard")
def dashboard():
    return render_template("index.html")


# =========================
# PREDICTION
# =========================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        slope = float(request.form.get("slope"))
        rainfall = float(request.form.get("rainfall"))
        temperature = float(request.form.get("temperature"))
        vibration = float(request.form.get("vibration"))
        crack = float(request.form.get("crack"))
        moisture = float(request.form.get("moisture"))
        displacement = float(request.form.get("displacement"))
        pressure = float(request.form.get("pressure"))

        features = [[
            slope,
            rainfall,
            temperature,
            vibration,
            crack,
            moisture,
            displacement,
            pressure
        ]]

        # Model prediction
        prediction = model.predict(features)[0]

        # Convert prediction to integer
        prediction = int(prediction)

        if prediction == 0:
            result = "SAFE"

        elif prediction == 1:
            result = "WARNING"

        else:
            result = "HIGH RISK"


        # =========================
        # REASONS
        # =========================

        reasons = []

        if slope > 60:
            reasons.append("High slope angle")

        if rainfall > 70:
            reasons.append("Heavy rainfall")

        if temperature > 35:
            reasons.append("High temperature")

        if vibration > 6:
            reasons.append("High vibration")

        if crack > 3:
            reasons.append("Large crack width")

        if moisture > 70:
            reasons.append("High soil moisture")

        if displacement > 2:
            reasons.append("Ground displacement")

        if pressure > 5:
            reasons.append("High pore pressure")

        if not reasons:
            reasons.append("All parameters are within normal limits.")


        # =========================
        # RECOMMENDATION
        # =========================

        if result == "SAFE":

            recommendation = (
                "Mining operations can continue safely. "
                "Continue regular monitoring."
            )

        elif result == "WARNING":

            recommendation = (
                "Increase monitoring and inspect the mining area."
            )

        else:

            recommendation = (
                "Stop mining immediately and inspect the rock face."
            )


        # =========================
        # TIME
        # =========================

        current_time = datetime.now().strftime(
            "%d-%m-%Y %I:%M %p"
        )


        # =========================
        # SAVE HISTORY
        # =========================

        file_exists = os.path.exists("history.csv")

        with open(
            "history.csv",
            "a",
            newline=""
        ) as file:

            writer = csv.writer(file)

            if not file_exists:

                writer.writerow([
                    "Slope",
                    "Rainfall",
                    "Temperature",
                    "Vibration",
                    "Crack",
                    "Moisture",
                    "Displacement",
                    "Pressure",
                    "Risk",
                    "Time"
                ])

            writer.writerow([
                slope,
                rainfall,
                temperature,
                vibration,
                crack,
                moisture,
                displacement,
                pressure,
                result,
                current_time
            ])


        # =========================
        # RESULT PAGE
        # =========================

        return render_template(
            "result.html",
            prediction=result,
            reasons=reasons,
            recommendation=recommendation,
            time=current_time
        )


    except Exception as e:

        return f"""
        <h2>Prediction Error</h2>
        <p>{str(e)}</p>
        <br>
        <a href="/dashboard">Go Back</a>
        """


# =========================
# HISTORY
# =========================

@app.route("/history")
def history():

    rows = []

    safe_count = 0
    warning_count = 0
    high_risk_count = 0

    if os.path.exists("history.csv"):

        with open(
            "history.csv",
            "r",
            newline=""
        ) as file:

            reader = csv.reader(file)
            rows = list(reader)

        risks = []

        for row in rows[1:]:

            if len(row) >= 9:
                risks.append(row[8])

        counts = Counter(risks)

        safe_count = counts.get("SAFE", 0)
        warning_count = counts.get("WARNING", 0)
        high_risk_count = counts.get("HIGH RISK", 0)


    return render_template(
        "history.html",
        rows=rows,
        safe_count=safe_count,
        warning_count=warning_count,
        high_risk_count=high_risk_count
    )


# =========================
# PDF REPORT
# =========================

@app.route("/download_report")
def download_report():

    filename = "report.pdf"

    pdf = canvas.Canvas(filename)

    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(
        150,
        800,
        "MineGuard AI Report"
    )

    pdf.setFont("Helvetica", 14)

    pdf.drawString(
        100,
        750,
        "Project: Rockfall Prediction System"
    )

    pdf.drawString(
        100,
        720,
        "Status: Active"
    )

    pdf.drawString(
        100,
        690,
        "Generated by MineGuard AI"
    )

    pdf.drawString(
        100,
        660,
        "Prediction history is stored in history.csv"
    )

    pdf.save()

    return send_file(
        filename,
        as_attachment=True
    )


# =========================
# MAP
# =========================

@app.route("/map")
def mine_map():

    return render_template("map.html")


# =========================
# RUN APPLICATION
# =========================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
