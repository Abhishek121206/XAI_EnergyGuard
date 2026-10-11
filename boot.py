import network
import time

SSID = "ISHANT"
PASSWORD = "123456789"

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