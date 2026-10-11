from flask import Flask, jsonify
from flask_cors import CORS
from pathlib import Path
import pandas as pd


# =====================================================
# APPLICATION
# =====================================================

app = Flask(__name__)
CORS(app)

# =====================================================
# PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

CLASSIFIED_FILE = DATA_DIR / "c1_classified.csv"
EXPLANATIONS_FILE = DATA_DIR / "c1_explanations.csv"


# =====================================================
# HELPER
# =====================================================

def load_csv(file_path):
    if not file_path.exists():
        return None

    return pd.read_csv(file_path)


# =====================================================
# HOME
# =====================================================

@app.route("/")
def home():
    return jsonify({
        "service": "XAI-EnergyGuard Backend",
        "status": "running"
    })


# =====================================================
# ALL CLASSIFIED DATA
# =====================================================

@app.route("/api/data", methods=["GET"])
def get_data():

    df = load_csv(CLASSIFIED_FILE)

    if df is None:
        return jsonify({
            "error": "Classified data file not found"
        }), 404

    return jsonify(
        df.to_dict(orient="records")
    )


# =====================================================
# ANOMALIES ONLY
# =====================================================

@app.route("/api/anomalies", methods=["GET"])
def get_anomalies():

    df = load_csv(CLASSIFIED_FILE)

    if df is None:
        return jsonify({
            "error": "Classified data file not found"
        }), 404

    anomalies = df[
        df["anomaly"] == "ANOMALY"
    ]

    return jsonify(
        anomalies.to_dict(
            orient="records"
        )
    )


# =====================================================
# EXPLANATIONS
# =====================================================

@app.route("/api/explanations", methods=["GET"])
def get_explanations():

    df = load_csv(EXPLANATIONS_FILE)

    if df is None:
        return jsonify({
            "error": "Explanation file not found"
        }), 404

    return jsonify(
        df.to_dict(
            orient="records"
        )
    )


# =====================================================
# LATEST ANOMALY
# =====================================================

@app.route("/api/anomalies/latest", methods=["GET"])
def get_latest_anomaly():

    df = load_csv(CLASSIFIED_FILE)

    if df is None:
        return jsonify({
            "error": "Classified data file not found"
        }), 404

    anomalies = df[
        df["anomaly"] == "ANOMALY"
    ]

    if anomalies.empty:
        return jsonify({
            "message": "No anomalies found"
        })

    latest = anomalies.iloc[-1]

    return jsonify(
        latest.to_dict()
    )


# =====================================================
# HEALTH CHECK
# =====================================================

@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy",
        "service": "XAI-EnergyGuard"
    })


# =====================================================
# RUN SERVER
# =====================================================

if __name__ == "__main__":

    print("======================================")
    print("XAI-EnergyGuard API")
    print("======================================")

    print("Starting Flask server...")
    print("URL: http://127.0.0.1:5000")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )