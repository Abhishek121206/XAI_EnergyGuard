the websiteimport json

import pandas as pd
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from ml_config import (
    CLASSES,
    CLASSIFIED_FILE,
    EXPLANATIONS_FILE,
    LATEST_PREDICTION_FILE,
    METRICS_FILE
)


# =====================================================
# APPLICATION
# =====================================================

app = Flask(__name__)

# Only the local dashboard may call the API
CORS(app, origins=[
    "http://127.0.0.1:5500",
    "http://localhost:5500"
])

DEFAULT_LIMITS = {
    "data": 500,
    "anomalies": 200,
    "explanations": 50
}


# =====================================================
# HELPERS
# =====================================================

def load_csv(file_path):

    if not file_path.exists():
        return None

    return pd.read_csv(file_path)


def load_json(file_path):

    if not file_path.exists():
        return None

    return json.loads(file_path.read_text())


def records(df):

    # pandas writes NaN as null (jsonify would emit invalid NaN)
    return Response(
        df.to_json(orient="records"),
        mimetype="application/json"
    )


def limit_param(name):

    """?limit=N (most recent N rows); ?limit=0 returns all."""

    limit = request.args.get(
        "limit",
        default=DEFAULT_LIMITS[name],
        type=int
    )

    return max(limit, 0)


def tail(df, limit):

    return df if limit == 0 else df.tail(limit)


def not_found(what):

    return jsonify({
        "error": f"{what} not found. "
                 "Run train.py and pipeline_runner.py."
    }), 404


def waste_events(df):

    return df[df["waste_type"] != "NORMAL"]


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
# CLASSIFIED DATA
# =====================================================

@app.route("/api/data", methods=["GET"])
def get_data():

    df = load_csv(CLASSIFIED_FILE)

    if df is None:
        return not_found("Classified data")

    return records(tail(df, limit_param("data")))


# =====================================================
# WASTE EVENTS
# =====================================================

@app.route("/api/anomalies", methods=["GET"])
def get_anomalies():

    df = load_csv(CLASSIFIED_FILE)

    if df is None:
        return not_found("Classified data")

    return records(
        tail(waste_events(df), limit_param("anomalies"))
    )


@app.route("/api/anomalies/latest", methods=["GET"])
def get_latest_anomaly():

    df = load_csv(CLASSIFIED_FILE)

    if df is None:
        return not_found("Classified data")

    anomalies = waste_events(df)

    if anomalies.empty:
        return jsonify({
            "message": "No anomalies found"
        })

    return Response(
        anomalies.iloc[-1].to_json(),
        mimetype="application/json"
    )


# =====================================================
# EXPLANATIONS
# =====================================================

@app.route("/api/explanations", methods=["GET"])
def get_explanations():

    df = load_csv(EXPLANATIONS_FILE)

    if df is None:
        return not_found("Explanations")

    return records(tail(df, limit_param("explanations")))


# =====================================================
# SUMMARY
# =====================================================

@app.route("/api/summary", methods=["GET"])
def get_summary():

    df = load_csv(CLASSIFIED_FILE)

    if df is None:
        return not_found("Classified data")

    counts = df["waste_type"].value_counts()

    metrics = load_json(METRICS_FILE) or {}

    return jsonify({
        "total_readings": int(len(df)),
        "class_counts": {
            label: int(counts.get(label, 0))
            for label in CLASSES
        },
        "anomaly_count": int(len(waste_events(df))),
        "outlier_count": int(df.get("is_outlier", pd.Series()).sum()),
        "first_timestamp": int(df["timestamp"].min()),
        "latest_timestamp": int(df["timestamp"].max()),
        "model_version": metrics.get("trained_at")
    })


# =====================================================
# MODEL METRICS
# =====================================================

@app.route("/api/model/metrics", methods=["GET"])
def get_model_metrics():

    metrics = load_json(METRICS_FILE)

    if metrics is None:
        return not_found("Model metrics")

    return jsonify(metrics)


# =====================================================
# LIVE PREDICTION (written by data_collector.py)
# =====================================================

@app.route("/api/live", methods=["GET"])
def get_live():

    prediction = load_json(LATEST_PREDICTION_FILE)

    if prediction is None:
        return jsonify({
            "error": "No live prediction yet. "
                     "Start data_collector.py with a trained model."
        }), 404

    return jsonify(prediction)


# =====================================================
# HEALTH CHECK
# =====================================================

@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy",
        "service": "XAI-EnergyGuard",
        "model_trained": METRICS_FILE.exists(),
        "pipeline_output": CLASSIFIED_FILE.exists()
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
