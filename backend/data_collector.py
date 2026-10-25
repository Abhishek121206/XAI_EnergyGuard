import csv
import json
import time
from pathlib import Path

import paho.mqtt.client as mqtt

from ml_config import LATEST_PREDICTION_FILE, MODEL_BUNDLE_FILE


# =====================================================
# MQTT CONFIGURATION
# =====================================================

MQTT_BROKER = "127.0.0.1"
MQTT_PORT = 1883
MQTT_TOPIC = "energyguard/+/reading"


# =====================================================
# DATA STORAGE
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

DATA_DIR.mkdir(exist_ok=True)


# =====================================================
# CSV COLUMNS
# =====================================================

CSV_FIELDS = [
    "timestamp",
    "circuit_id",
    "distance_cm",
    "presence",
    "temperature_C",
    "humidity_percent",
    "voltage_V",
    "current_A",
    "power_W",
    "status"
]


# =====================================================
# LIVE PREDICTION
# =====================================================

# One Predictor per circuit (each keeps its own short
# reading history). Disabled if no model is trained.

predictors = {}

live_prediction_enabled = MODEL_BUNDLE_FILE.exists()


def predict_live(circuit_id, row):

    global live_prediction_enabled

    if not live_prediction_enabled:
        return None

    try:

        if circuit_id not in predictors:

            from predict import Predictor

            predictors[circuit_id] = Predictor()

        prediction = predictors[circuit_id].predict_reading(row)

        if prediction is None:
            return None

        # Write then rename so the API never reads a half file
        temp_file = LATEST_PREDICTION_FILE.with_suffix(".tmp")
        temp_file.write_text(json.dumps(prediction))
        temp_file.replace(LATEST_PREDICTION_FILE)

        return prediction

    except Exception as error:

        print("Live prediction disabled:", error)

        live_prediction_enabled = False

        return None


# =====================================================
# MQTT CONNECT
# =====================================================

def on_connect(client, userdata, flags, reason_code, properties):

    if reason_code == 0:

        print("MQTT connected")
        print("Subscribing to:", MQTT_TOPIC)

        client.subscribe(MQTT_TOPIC)

    else:

        print("MQTT connection failed")
        print("Reason:", reason_code)


# =====================================================
# MQTT MESSAGE
# =====================================================

def on_message(client, userdata, message):

    try:

        payload = message.payload.decode("utf-8")

        data = json.loads(payload)

    except Exception as error:

        print("Invalid MQTT message:", error)

        return


    # -------------------------------------------------
    # CIRCUIT ID FROM MQTT TOPIC
    #
    # energyguard/c1/reading
    #              ↑
    # -------------------------------------------------

    topic_parts = message.topic.split("/")

    if len(topic_parts) != 3:

        print("Invalid topic:", message.topic)

        return

    circuit_id = topic_parts[1]


    # -------------------------------------------------
    # BACKEND TIMESTAMP
    # -------------------------------------------------

    timestamp = int(time.time())


    # -------------------------------------------------
    # CREATE CSV ROW
    # -------------------------------------------------

    row = {

        "timestamp":
            timestamp,

        "circuit_id":
            circuit_id,

        "distance_cm":
            data.get("distance_cm"),

        "presence":
            data.get("presence"),

        "temperature_C":
            data.get("temperature_C"),

        "humidity_percent":
            data.get("humidity_percent"),

        "voltage_V":
            data.get("voltage_V"),

        "current_A":
            data.get("current_A"),

        "power_W":
            data.get("power_W"),

        "status":
            data.get("status")
    }


    # -------------------------------------------------
    # CSV FILE
    # -------------------------------------------------

    csv_file = DATA_DIR / f"{circuit_id}.csv"

    file_exists = csv_file.exists()


    try:

        with csv_file.open(
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=CSV_FIELDS
            )

            if not file_exists:

                writer.writeheader()

            writer.writerow(row)


        print()
        print("========== MQTT DATA ==========")
        print("Topic :", message.topic)
        print("Circuit:", circuit_id)
        print("Power :", row["power_W"], "W")
        print("Presence:", row["presence"])
        print("Status :", row["status"])

        prediction = predict_live(circuit_id, row)

        if prediction is not None:
            print(
                "Model  :",
                prediction["waste_type"],
                f"({prediction['waste_confidence']:.0%})"
            )

        print("Saved :", csv_file)
        print("===============================")


    except OSError as error:

        print("CSV error:", error)


# =====================================================
# MQTT CLIENT
# =====================================================

client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2,
    client_id="energyguard_data_collector"
)

client.on_connect = on_connect
client.on_message = on_message


# =====================================================
# START
# =====================================================

print("======================================")
print("XAI-EnergyGuard DATA COLLECTOR")
print("======================================")

print("Broker :", MQTT_BROKER)
print("Port   :", MQTT_PORT)
print("Topic  :", MQTT_TOPIC)
print(
    "Live ML:",
    "enabled" if live_prediction_enabled
    else "disabled (run train.py)"
)

print()
print("Connecting to MQTT broker...")

client.connect(
    MQTT_BROKER,
    MQTT_PORT,
    60
)

client.loop_forever()