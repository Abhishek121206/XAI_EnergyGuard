# XAI-EnergyGuard

Detects and **explains** wasted electricity in a room using an ESP32 sensor node, an XGBoost classifier, SHAP, DiCE counterfactuals and plain-language explanations for three audiences.

```
ESP32 (main.py) --MQTT--> Mosquitto --> data_collector.py --> data/c1.csv
                                              |  (live prediction per reading)
                                              v
                                   data/latest_prediction.json

pipeline_runner.py:  feature_engineer -> predict -> xai_module -> dice_module -> nlg_module
                                                      (SHAP)       (DiCE)        (text)
app.py (Flask API :5000)  -->  frontend/ (dashboard :5500)
```

## Classes

| Code | Meaning | Label source |
|---|---|---|
| NORMAL | Expected use | Firmware status `NORMAL` / `NO_LOAD` |
| IE | Idle equipment: load on, nobody present | Firmware status `IDLE_WASTE` |
| OC | Overconsumption: occupied, above 1.5 W | Firmware status `OVERCONSUMPTION` |
| UP | Unusual pattern: occupied, sudden power change > max(3× recent spread, 1 W) | Rule in `features.py` |

> **Important:** IE/OC/NORMAL labels come from the ESP32 firmware rules, so high scores for those classes mean the model reproduces those rules, not independent ground truth. To get stronger evidence, record controlled sessions with known labels (e.g. "lamp on, room empty, 10 min") and add them to the training data.

## Hardware

ESP32 + DHT11 (GPIO 4), HC-SR04 (TRIG 27 / ECHO 26), ACS712 (GPIO 35), green LED 18, red LED 23, buzzer 19. Presence = object within 100 cm.

## Setup (once)

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

ESP32: copy `device_config_example.py` to `device_config.py`, fill in your hotspot name, password and laptop IP, then upload `boot.py`, `main.py` and `device_config.py` with Thonny.

Mosquitto runs as a Windows service: `Get-Service mosquitto` (start with `Start-Service mosquitto`).

## Train the model

```powershell
cd backend
python train.py
```

This trains on the **real** readings in `data/c1.csv` and:

- splits chronologically (70 / 15 / 15 %), so the test set is the most recent data the model never saw
- computes the hourly power baseline from training data only
- trains XGBoost (class-balanced, early stopping) and an Isolation Forest (outlier score)
- compares against the firmware rule and logistic regression
- saves `models/model_bundle.joblib` and `models/metrics.json`

Re-run it whenever you've collected meaningfully more data.

## Run

One command (opens one window per service, then the dashboard):

```powershell
.\run_all.ps1                # collector + live pipeline + API + dashboard
.\run_all.ps1 -NoCollector   # demo from saved data, no ESP32 needed
.\run_all.ps1 -Train         # retrain first
```

Or manually, each in its own terminal from `backend/`:

| Command | Purpose |
|---|---|
| `python data_collector.py` | Saves MQTT readings to `data/c1.csv` and classifies each one live |
| `python live_pipeline.py` | Re-runs the full XAI pipeline every 30 s when new data arrives |
| `python app.py` | API on http://127.0.0.1:5000 |
| `python -m http.server 5500` (in `frontend/`) | Dashboard on http://127.0.0.1:5500 |

## API

| Endpoint | Returns |
|---|---|
| `GET /api/health` | Service status, whether a model is trained |
| `GET /api/summary` | Reading count, per-class counts, outliers |
| `GET /api/data?limit=N` | Latest N classified readings (0 = all) |
| `GET /api/anomalies?limit=N` | Latest N waste events (IE / OC / UP) |
| `GET /api/anomalies/latest` | Most recent waste event |
| `GET /api/explanations?limit=N` | Latest N explanations (manager / student / admin + counterfactual) |
| `GET /api/model/metrics` | Test-set metrics, baselines, feature importance |
| `GET /api/live` | Latest live prediction from the collector |

## Tests

```powershell
cd backend
python -m pytest tests
```

## Files

| Path | Role |
|---|---|
| `boot.py`, `main.py` | ESP32 firmware (MicroPython) |
| `backend/ml_config.py` | Paths, classes, features, thresholds |
| `backend/features.py` | Feature engineering + labels (shared by training, pipeline and live) |
| `backend/train.py` | Model training and evaluation |
| `backend/predict.py` | `Predictor` for batch and live inference |
| `backend/xai_module.py` | SHAP top-3 reasons per waste event (cached per model version) |
| `backend/dice_module.py` | Counterfactuals: DiCE on the latest events, model search for the rest, rule fallback |
| `backend/nlg_module.py` | Audience-specific explanations |
