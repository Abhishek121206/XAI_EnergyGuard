Yep bro. Since you closed everything, use this exact restart sequence.

## 1. Turn ON laptop hotspot

Windows:

**Settings → Network & Internet → Mobile hotspot → ON**

Make sure the hotspot name is:

```text
ISHANT
```

## 2. Start ESP32

Open **Thonny** and connect the ESP32.

Run `boot.py`.

You should get:

```text
WiFi connected
ESP32 IP: 192.168.137.x
```

Then run `main.py`.

You should see:

```text
MQTT connected
MQTT : PUBLISHED
DATA: {...}
```

Keep Thonny open.

## 3. Mosquitto

You **do not start Mosquitto manually**.

It is installed as a Windows service and should already be running.

Check in PowerShell:

```powershell
Get-Service mosquitto
```

You want:

```text
Running  mosquitto
```

If it isn't running:

```powershell
Start-Service mosquitto
```

## 4. Start the Python data collector

Open a new PowerShell.

Run:

```powershell
cd "C:\esp32_with_micropython\Vnest\XAI-EnergyGuard\backend"
```

Activate the environment:

```powershell
.\.venv\Scripts\activate
```

Then:

```powershell
python data_collector.py
```

You should see:

```text
MQTT connected
Subscribing to: energyguard/+/reading
```

Keep this terminal open.

## 5. Generate the current ML/XAI results

Open another PowerShell.

Run:

```powershell
cd "C:\esp32_with_micropython\Vnest\XAI-EnergyGuard\backend"
.\.venv\Scripts\activate
python pipeline_runner.py
```

Wait until:

```text
FULL PIPELINE COMPLETE
```

This updates:

```text
c1_features.csv
c1_anomalies.csv
c1_classified.csv
c1_shap.csv
c1_counterfactuals.csv
c1_explanations.csv
```

## 6. Start Flask API

In another PowerShell:

```powershell
cd "C:\esp32_with_micropython\Vnest\XAI-EnergyGuard\backend"
.\.venv\Scripts\activate
python app.py
```

Keep it running.

You should see:

```text
XAI-EnergyGuard API
Starting Flask server...
URL: http://127.0.0.1:5000
```

## 7. Start frontend

Open another PowerShell:

```powershell
cd "C:\esp32_with_micropython\Vnest\XAI-EnergyGuard\frontend"
python -m http.server 5500
```

Keep it running.

## 8. Open dashboard

Go to:

```text
http://127.0.0.1:5500
```

You should now see:

```text
Energy Overview
Current Power
Presence
Total Readings
Anomalies
Latest Reading
Latest Anomaly
Recent Readings
AI Explanations
```

### Your terminals should finally look like this

```text
Thonny
└── ESP32 main.py running

PowerShell 1
└── data_collector.py

PowerShell 2
└── pipeline_runner.py

PowerShell 3
└── app.py

PowerShell 4
└── frontend http.server :5500

Mosquitto
└── Windows background service
```

You **do not need `mosquitto_sub`** for the demo anymore unless you specifically want to watch MQTT messages.

One limitation in our current version: the dashboard updates from the CSV/API, but the full ML pipeline is still manually triggered with `pipeline_runner.py`. So for a live demo, run `pipeline_runner.py` after collecting some new readings.