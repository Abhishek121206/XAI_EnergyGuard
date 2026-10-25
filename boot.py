import network
import time

from device_config import WIFI_SSID as SSID
from device_config import WIFI_PASSWORD as PASSWORD

wifi = network.WLAN(network.STA_IF)
wifi.active(True)

print("Connecting to WiFi...")

wifi.connect(SSID, PASSWORD)

for i in range(20):
    if wifi.isconnected():
        break

    print("Waiting...", i + 1)
    time.sleep(1)

if wifi.isconnected():
    print("WiFi connected")
    print("ESP32 IP:", wifi.ifconfig()[0])
    print("Network config:", wifi.ifconfig())
else:
    print("WiFi connection FAILED")
    print("Status:", wifi.status())