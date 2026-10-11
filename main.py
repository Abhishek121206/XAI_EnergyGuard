from machine import Pin, ADC
import dht
import time
import json
from umqtt.simple import MQTTClient


# =====================================================
# MQTT CONFIGURATION
# =====================================================

MQTT_BROKER = "192.168.137.1"
MQTT_PORT = 1883

MQTT_CLIENT_ID = "esp32_energyguard"
MQTT_TOPIC = b"energyguard/c1/reading"


# =====================================================
# PIN CONFIGURATION
# =====================================================

DHT_PIN = 4

TRIG_PIN = 27
ECHO_PIN = 26

ACS_PIN = 35

GREEN_LED_PIN = 18
RED_LED_PIN = 23

BUZZER_PIN = 19


# =====================================================
# SETTINGS
# =====================================================

PRESENCE_LIMIT = 100.0       # cm

LOAD_VOLTAGE = 5.0           # volts

ACS_SENSITIVITY = 0.066      # V/A

CURRENT_NOISE_LIMIT = 0.02   # A


# =====================================================
# INITIALIZE COMPONENTS
# =====================================================

dht_sensor = dht.DHT11(Pin(DHT_PIN))

trig = Pin(TRIG_PIN, Pin.OUT)
echo = Pin(ECHO_PIN, Pin.IN)

adc = ADC(Pin(ACS_PIN))
adc.atten(ADC.ATTN_11DB)

green = Pin(GREEN_LED_PIN, Pin.OUT)
red = Pin(RED_LED_PIN, Pin.OUT)

buzzer = Pin(BUZZER_PIN, Pin.OUT)


# =====================================================
# INITIAL STATE
# =====================================================

green.off()
red.off()
buzzer.off()


# =====================================================
# ACS712 ZERO CALIBRATION
# =====================================================

print()
print("======================================")
print("ACS712 CALIBRATION")
print("======================================")
print("KEEP LOAD OFF")
print("Please wait...")
print()

total = 0

for i in range(100):

    total += adc.read()

    time.sleep_ms(2)

ZERO_ADC = total / 100

print("Zero ADC =", ZERO_ADC)
print("Calibration complete.")
print("SYSTEM STARTING...")
print()


# =====================================================
# MQTT CONNECTION
# =====================================================

print("Connecting to MQTT broker...")

mqtt = MQTTClient(
    MQTT_CLIENT_ID,
    MQTT_BROKER,
    port=MQTT_PORT
)

mqtt.connect()

print("MQTT connected")
print("MQTT topic:", MQTT_TOPIC.decode())


# =====================================================
# ULTRASONIC DISTANCE
# =====================================================

def get_distance():

    trig.value(0)
    time.sleep_us(2)

    trig.value(1)
    time.sleep_us(10)

    trig.value(0)

    # Wait for echo HIGH

    timeout = time.ticks_us()

    while echo.value() == 0:

        if time.ticks_diff(
            time.ticks_us(),
            timeout
        ) > 30000:

            return None

    start = time.ticks_us()

    # Wait for echo LOW

    while echo.value() == 1:

        if time.ticks_diff(
            time.ticks_us(),
            start
        ) > 30000:

            return None

    end = time.ticks_us()

    duration = time.ticks_diff(
        end,
        start
    )

    distance = (
        duration * 0.0343
    ) / 2

    return distance


# =====================================================
# ACS712 CURRENT
# =====================================================

def get_current():

    total = 0

    samples = 20

    for i in range(samples):

        total += adc.read()

        time.sleep_ms(1)

    average_adc = total / samples

    adc_difference = (
        average_adc - ZERO_ADC
    )

    gpio_voltage_difference = (
        adc_difference / 4095
    ) * 3.3

    sensor_voltage_difference = (
        gpio_voltage_difference * 2
    )

    current = (
        sensor_voltage_difference /
        ACS_SENSITIVITY
    )

    if abs(current) < CURRENT_NOISE_LIMIT:

        current = 0.0

    current = abs(current)

    return current


# =====================================================
# DHT11
# =====================================================

def get_temperature_humidity():

    try:

        dht_sensor.measure()

        temperature = dht_sensor.temperature()

        humidity = dht_sensor.humidity()

        return temperature, humidity

    except Exception as e:

        print("DHT11 error:", e)

        return None, None


# =====================================================
# MAIN LOOP
# =====================================================

while True:

    # -------------------------------------------------
    # DISTANCE
    # -------------------------------------------------

    distance = get_distance()


    # -------------------------------------------------
    # PRESENCE
    # -------------------------------------------------

    if (
        distance is not None
        and
        distance <= PRESENCE_LIMIT
    ):

        presence = "PRESENT"

        green.on()
        red.off()

    else:

        presence = "ABSENT"

        green.off()
        red.on()


    # -------------------------------------------------
    # TEMPERATURE + HUMIDITY
    # -------------------------------------------------

    temperature, humidity = (
        get_temperature_humidity()
    )


    # -------------------------------------------------
    # CURRENT
    # -------------------------------------------------

    current = get_current()


    # -------------------------------------------------
    # VOLTAGE
    # -------------------------------------------------

    voltage = LOAD_VOLTAGE


    # -------------------------------------------------
    # POWER
    # -------------------------------------------------

    power = voltage * current


    # -------------------------------------------------
    # STATUS
    # -------------------------------------------------

    if (
        presence == "ABSENT"
        and
        current > CURRENT_NOISE_LIMIT
    ):

        status = "IDLE_WASTE"

        buzzer.on()

    elif power > 1.50:

        status = "OVERCONSUMPTION"

        buzzer.off()

    elif presence == "PRESENT":

        status = "NORMAL"

        buzzer.off()

    else:

        status = "NO_LOAD"

        buzzer.off()


    # =================================================
    # JSON DATA
    # =================================================

    data = {

        "distance_cm":
            round(distance, 1)
            if distance is not None
            else None,

        "presence":
            presence,

        "temperature_C":
            temperature,

        "humidity_percent":
            humidity,

        "voltage_V":
            round(voltage, 2),

        "current_A":
            round(current, 3),

        "power_W":
            round(power, 3),

        "status":
            status
    }


    # =================================================
    # MQTT PUBLISH
    # =================================================

    message = json.dumps(data)

    try:

        mqtt.publish(
            MQTT_TOPIC,
            message.encode()
        )

        mqtt_status = "PUBLISHED"

    except Exception as e:

        mqtt_status = "ERROR: " + str(e)


    # =================================================
    # SERIAL OUTPUT
    # =================================================

    print()
    print("======================================")

    if distance is not None:

        print(
            "Distance   :",
            round(distance, 1),
            "cm"
        )

    else:

        print(
            "Distance   : OUT OF RANGE"
        )

    print(
        "Presence   :",
        presence
    )

    if temperature is not None:

        print(
            "Temperature:",
            temperature,
            "C"
        )

        print(
            "Humidity   :",
            humidity,
            "%"
        )

    else:

        print(
            "Temperature: ERROR"
        )

    print(
        "Voltage    :",
        round(voltage, 2),
        "V"
    )

    print(
        "Current    :",
        round(current, 3),
        "A"
    )

    print(
        "Power      :",
        round(power, 3),
        "W"
    )

    print(
        "STATUS     :",
        status
    )

    print(
        "MQTT       :",
        mqtt_status
    )

    print("======================================")

    print(
        "DATA:",
        message
    )

    time.sleep_ms(500)
