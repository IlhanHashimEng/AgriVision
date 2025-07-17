import requests
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import time
from pyzbar.pyzbar import decode
import pytz
from PIL import Image
import numpy as np
from pycoral.adapters import classify, common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
import minimalmodbus
import serial
from pymodbus.client import ModbusTcpClient
import RPi.GPIO as GPIO
import ast
import cohere


# Setup for Cohere
co = cohere.Client("OJw8N3ssYQpQi2FS2mmZXSVDeJiP06IhvkAVkfAU")



# Wise-Paas Datahub Packages
from wisepaasdatahubedgesdk.EdgeAgent import EdgeAgent
import wisepaasdatahubedgesdk.Common.Constants as constant
from wisepaasdatahubedgesdk.Model.Edge import EdgeAgentOptions, DCCSOptions, EdgeData, EdgeTag

API_KEY = os.getenv("OPENWEATHER_API_KEY", "792ca464e033a3d0e5e009204cd23a51")
WEATHER_API_URL = "https://api.openweathermap.org/data/2.5/forecast"
LOCATION = "Nibong Tebal"
LOG_FILE = "system.log"
MAX_RETRIES = 5
BUFFER_FOLDER = "./bufferx"
MODEL_FILE = "mobilenet_v2_1.0_224_quant_edgetpu.tflite"
LABEL_FILE = "nutrients_lettuce.txt"

ADAM_IP = "10.0.0.1"
ADAM_PORT = 502
HUMIDITY_REGISTER = 2
UNIT_ID = 1
OPT_TIME = 0.3
MAX_REGISTER_VALUE = 65535 # Ilhan : Added this

THRESHOLDS = {
    # This in mg/kg value
    "N": 75,  # Minimum acceptable level for nitrogen
    "P": 15,  # Minimum acceptable level for phosphorus
    "K": 150,  # Minimum acceptable level for potassium
    "humidity": 70,  # Minimum acceptable soil humidity level
    "temperature": 18,  # Minimum temperature for watering
    "pH": 6  #Minimum pH for soil pH level
}

relay_pins = [16, 18, 27, 22, 23, 24, 25, 21, 20]

relay_mapping = {
    # NOTE THIS IS THE RELAY LIST
    "Solution A": 1,       # Relay 1 controls Solution A motor
    "Solution B": 2,       # Relay 2 controls Solution B motor
    "Main Water": 3,       # Relay 3 controls the main water motor10
    "Mixing Tank": 4,      # Relay 4 controls the mixing tank motor
    "Sol1": 5,             # Relay 5 controls Solenoid 1
    "Sol2": 6,             # Relay 6 controls Solenoid 2
    "Sol3": 7,             # Relay 7 controls the plant watering motor
    "Sol4": 8,             # Relay 8 controls Solenoid 4 for mixing
    "Sol5": 9              # Ilhan : Relay 9 controls Solenoid 5
}

# Wise-PaaS setup
edgeAgentOptions = EdgeAgentOptions(nodeId="8b621887-6564-4b08-a535-9778b3c928c5")
edgeAgentOptions.connectType = constant.ConnectType["DCCS"]
dccsOptions = DCCSOptions(
    apiUrl="https://api-dccs-ensaas.education.wise-paas.com/",
    credentialKey="23577ec4a6ab06cea0b3b37dcc631624"
)
edgeAgentOptions.DCCS = dccsOptions
_edgeAgent = EdgeAgent(edgeAgentOptions)

def setup_relays():
    """Set up GPIO pins for relays."""
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    for pin in relay_pins:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.HIGH)

def activate_relay(channel):
    """Turn on the relay for the given channel (1-8)."""
    GPIO.output(relay_pins[channel - 1], GPIO.LOW)

def deactivate_relay(channel):
    """Turn off the relay for the given channel (1-8)."""
    GPIO.output(relay_pins[channel - 1], GPIO.HIGH)

def cleanup():
    """Clean up GPIO settings."""
    GPIO.cleanup()

def log_message(message):
    """Logs messages to a log file with a timestamp."""
    with open(LOG_FILE, "a") as log:
        log.write(f"{datetime.now()} >->> {message}\n")
    print(message)


def fetch_forecast_data(api_key, location):
    """Fetch weather forecast data from the OpenWeather API."""
    url = f"{WEATHER_API_URL}?appid={api_key}&q={location}&units=metric"
    try:
        response = requests.get(url)
        response.raise_for_status()
        log_message("Successfully fetched weather forecast data.")
        return response.json()
    except requests.exceptions.RequestException as e:
        log_message(f"Error fetching forecast data: {e}")
        return None

def process_forecast_data(response):
    """Filter forecast data for the next 24 hours."""
    local_tz = pytz.timezone("Asia/Kuala_Lumpur")
    utc_tz = pytz.utc

    current_time_local = datetime.now(local_tz)
    end_time_local = current_time_local + timedelta(hours=27)
    # print(f"Here is the current time: {current_time_local}")
    # print(f"Here is the end time: {end_time_local}")
    if response.get("cod") == "200":
        forecast_data = []
        for forecast in response["list"]:
            forecast_time_utc = datetime.fromtimestamp(forecast["dt"], tz=utc_tz)
            forecast_time_local = forecast_time_utc.astimezone(local_tz)
            # print(f"Here is the forecast time: {forecast_time_local}")
            if current_time_local <= forecast_time_local < end_time_local:
                forecast_data.append({
                    "time": forecast_time_local.strftime('%Y-%m-%d %H:%M:%S'),
                    "temp": forecast["main"]["temp"],
                    "humidity": forecast["main"]["humidity"],
                    "description": forecast["weather"][0]["description"],
                    "rain_prob": forecast.get("pop", 0) * 100,
                    "rain_volume": forecast.get("rain", {}).get("3h", 0)
                })
        log_message("Processed forecast data for the next 24 hours.")
        return forecast_data
    else:
        log_message(f"Error in forecast data response: {response.get('message')}")
        return None

def calculate_derivative(forecast_data):
    """Calculate temperature and humidity derivatives"""
    for i in range(len(forecast_data)):
        if i == 0:
            forecast_data[i]["temp_derivative"] = 0
            forecast_data[i]["humidity_derivative"] = 0
        else:
            forecast_data[i]["temp_derivative"] = forecast_data[i]["temp"] - forecast_data[i - 1]["temp"]
            forecast_data[i]["humidity_derivative"] = forecast_data[i]["humidity"] - forecast_data[i - 1]["humidity"]
    log_message("Calculated derivatives for forecast data.")
    return forecast_data

def scheduled_watering_with_derivative(forecast_data, soil_moisture_data=None):
    """
    Determine watering schedule based on forecast data, derivatives, and soil moisture
    - Includes dynamic thresholds for temperature, humidity, and rain probability
    - Uses soil moisture data for additional decision-making
    - Get soil moisture data from humidity sensor from past day to ensure that the data is transferable
    """
    watering_schedule = []
    soil_moisture_data = soil_moisture_data or {}  # Default to an empty dictionary if no data provided

    for i in range(len(forecast_data)):
        temp = forecast_data[i]["temp"]
        humidity = forecast_data[i]["humidity"]
        rain_prob = forecast_data[i]["rain_prob"]
        soil_moisture = soil_moisture_data.get(forecast_data[i]["time"], 30)  # Default to 30% if not provided to assume dry conditions

        temp_threshold = 25 if rain_prob < 30 else 28
        humidity_threshold = 60 if rain_prob < 30 else 70
        soil_moisture_threshold = 40  # Avoid watering if soil moisture > 40%

        if i < 0:
            forecast_data[i]["water"] = "No"
            forecast_data[i]["reason"] = "Not enough data for analysis."
        else:
            # Determine if watering is needed based on thresholds and derivatives
            if (
                temp > temp_threshold and
                humidity < humidity_threshold and
                rain_prob < 50 and
                soil_moisture < soil_moisture_threshold and
                forecast_data[i]["temp_derivative"] >= 0 and
                forecast_data[i]["humidity_derivative"] <= 0
            ):
                forecast_data[i]["water"] = "Yes"
                forecast_data[i]["reason"] = (
                    f"Temperature and Humidity thresholds met for 3 hours with non-negative derivatives. "
                    f"Soil Moisture: {soil_moisture}% < {soil_moisture_threshold}%. "
                    f"Rain Prob: {rain_prob}% < 50%."
                )
            else:
                forecast_data[i]["water"] = "No"
                forecast_data[i]["reason"] = (
                    f"Conditions not met. Temp: {temp}°C, Humidity: {humidity}%, "
                    f"Soil Moisture: {soil_moisture}%, Rain Prob: {rain_prob}%."
                )

        watering_schedule.append(forecast_data[i])

    log_message("Watering schedule generated based on forecast data and soil moisture.")
    return watering_schedule



def send_watering_schedule_to_wisepaas(watering_schedule):
    """Send watering schedule to Wise-PaaS."""
    for entry in watering_schedule:
        edgeData = EdgeData()
        edgeData.tagList.append(EdgeTag(deviceId="WeatherStation", tagName="Time", value=entry["time"]))
        edgeData.tagList.append(EdgeTag(deviceId="WeatherStation", tagName="Temp", value=entry["temp"]))
        edgeData.tagList.append(EdgeTag(deviceId="WeatherStation", tagName="Humidity", value=entry["humidity"]))
        edgeData.tagList.append(EdgeTag(deviceId="WeatherStation", tagName="RainProb", value=entry["rain_prob"]))
        edgeData.tagList.append(EdgeTag(deviceId="WeatherStation", tagName="Description", value=entry["description"]))
        edgeData.tagList.append(EdgeTag(deviceId="WeatherStation", tagName="Water", value=entry["water"]))
        edgeData.tagList.append(EdgeTag(deviceId="WeatherStation", tagName="Reason", value=entry["reason"]))

        edgeData.timestamp = datetime.now()
        _edgeAgent.sendData(edgeData)

        log_message(
            f"Sent data for time: {entry['time']}, Temp: {entry['temp']}°C, "
            f"Humidity: {entry['humidity']}%, RainProb: {entry['rain_prob']}%, "
            f"Description: {entry['description']}, Water: {entry['water']}, "
            f"Reason: {entry['reason']}"
        )
        time.sleep(3)

def decode_qr_code(filename):
    """
    Decode a QR code from an image file.

    Args:
        filename (str): The filename of the QR code image to decode.

    Returns:
        dict: A dictionary containing sensor values or an error message.
    """
    try:
        # Open the image
        img = Image.open(filename)
        decoded_objects = decode(img)

        if decoded_objects:
            qr_data = decoded_objects[0].data.decode("utf-8")
            print(f"Decoded QR data: {qr_data}")

            # If QR data is already a dictionary-like string, safely evaluate it
            try:
                parsed_data = ast.literal_eval(qr_data)
                if isinstance(parsed_data, dict):
                    return parsed_data
                else:
                    return {"error": "QR code does not contain a valid dictionary."}
            except Exception as eval_error:
                return {"error": f"Error parsing QR code data: {eval_error}"}

        else:
            return {"error": "No QR code found"}

    except FileNotFoundError:
        return {"error": f"File not found: {filename}"}
    except Exception as e:
        return {"error": str(e)}

def plot_forecast(forecast_df, filename="forecast_plot.png"):
    """
    Visualize forecast data and watering schedule, and save the plot as a PNG file
    """
    plt.figure(figsize=(15, 8))

    plt.plot(forecast_df["time"], forecast_df["temp"], label="Temperature (°C)", marker="o")
    plt.plot(forecast_df["time"], forecast_df["rain_prob"], label="Rain Probability (%)", marker="x")
    plt.plot(forecast_df["time"], forecast_df["humidity"], label="Humidity (%)", marker="s")

    watering_indices = forecast_df[forecast_df["water"] == "Yes"].index
    plt.scatter(
        forecast_df.loc[watering_indices, "time"],
        [0] * len(watering_indices),
        color="red", label="Scheduled Watering", zorder=5, marker="o", s=50
    )

    plt.style.use('dark_background')
    plt.xlabel("Date Time")
    plt.ylabel("Metrics")
    plt.title(f"Smart Watering Schedule at {LOCATION}")
    plt.xticks(rotation=90)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(filename)
    print(f"Plot saved as {filename}")

    plt.show()


def get_dummy_values():
    """Return dummy values when sensors are unavailable."""
    return {"N": 999, "P": 999, "K": 999, "humidity": 999, "pH": 999}


def fetch_sensor_data():
    """
    Fetch data from a decoded QR code.
    Returns sensor data directly from the QR code in the specified format.
    """
    data = {
        "N": 999,
        "P": 999,
        "K": 999,
        "humidity": 999,
        "pH": 999
    }

    try:
        # Locate QR code image file
        png_files = [f for f in os.listdir(BUFFER_FOLDER) if f.endswith(".png")]
        if not png_files:
            log_message("No QR code image found in the buffer folder. Using dummy values.")
        else:
            qr_code_file = os.path.join(BUFFER_FOLDER, png_files[0])
            qr_data = decode_qr_code(qr_code_file)

            if qr_data.get("error"):
                log_message(f"Error decoding QR code: {qr_data['error']}. Using dummy values.")
            else:
                # Update data with values from the QR code
                data.update(qr_data)

    except Exception as e:
        log_message(f"Unexpected error while processing QR code: {e}. Using dummy values.")

    log_message(f"Final sensor data: {data}")

    return data

def analyze_crop_data(data):
    prompt = f"""
    Provide actionable feedback to a farmer based on the following data:
    - Nitrogen: {data['N']} ppm
    - Phosphorus: {data['P']} ppm
    - Potassium: {data['K']} ppm
    - Humidity: {data['humidity']} %
    - pH: {data['pH']}
    Write it an email format including values of each data
    """
    response = co.generate(
        model="command-xlarge",
        prompt=prompt,
        max_tokens=150
    )
    return response.generations[0].text.strip()

# def fetch_sensor_data():
#     """
#     Fetch data from sensors with improved modular error handling.
#     Returns sensor data with dummy values only for sensors that fail.
#     """
#     data = {
#         "N": 999,
#         "P": 999,
#         "K": 999,
#         "humidity": 999,
#         "pH": 999
#     }

#     try:


#         # Initialize NPK sensor
#         NPK_sensor = minimalmodbus.Instrument('/dev/ttyUSB0', 1)
#         NPK_sensor.serial.baudrate = 4800
#         NPK_sensor.serial.bytesize = 8
#         NPK_sensor.serial.parity = serial.PARITY_NONE
#         NPK_sensor.serial.stopbits = 1
#         NPK_sensor.serial.timeout = 1
#         NPK_sensor.mode = minimalmodbus.MODE_RTU

#         # Try reading nitrogen
#         try:
#             data["N"] = NPK_sensor.read_register(30, 0, functioncode=3)
#         except minimalmodbus.NoResponseError:
#             log_message("Failed to read nitrogen. Using dummy value.")

#         # Try reading phosphorus
#         try:
#             data["P"] = NPK_sensor.read_register(31, 0, functioncode=3)
#         except minimalmodbus.NoResponseError:
#             log_message("Failed to read phosphorus. Using dummy value.")

#         # Try reading potassium
#         try:
#             data["K"] = NPK_sensor.read_register(32, 0, functioncode=3)
#         except minimalmodbus.NoResponseError:
#             log_message("Failed to read potassium. Using dummy value.")

#     except serial.SerialException as e:
#         log_message(f"Error initializing NPK sensor: {e}. Using dummy values for N, P, K.")

#     try:
#         # connect to ADAM for humidity
#         adam_client = ModbusTcpClient(ADAM_IP, port=ADAM_PORT, timeout=15)
#         if adam_client.connect():
#             try:
#                 humidity_result = adam_client.read_holding_registers(HUMIDITY_REGISTER, count=1, slave=UNIT_ID)
#                 if not humidity_result.isError():
#                     data["humidity"] = convert_to_inverted_percentage(humidity_result.registers[0])
#                 else:
#                     log_message("Error reading humidity from ADAM-6217. Using dummy value.")
#             except Exception as e:
#                 log_message(f"Error reading humidity from ADAM-6217: {e}. Using dummy value.")
#             finally:
#                 adam_client.close()
#         else:
#             log_message("Failed to connect to ADAM-6217. Using dummy value for humidity.")
#     except Exception as e:
#         log_message(f"Unexpected error with ADAM sensor: {e}. Using dummy value for humidity.")

#     # Decode pH value from QR code
#     try:
#         png_files = [f for f in os.listdir(BUFFER_FOLDER) if f.endswith(".png")]
#         if not png_files:
#             log_message("No QR code image found in the buffer folder. Using dummy value for pH.")
#             data["pH"] = 7  # Default to neutral pH
#         else:
#             qr_code_file = os.path.join(BUFFER_FOLDER, png_files[0])
#             qr_data = decode_qr_code(qr_code_file)
#             print(f"Here is the pH value" + str(qr_data))
#             data["pH"] = qr_data.get("pH", 7)
#     except Exception as e:
#         log_message(f"Error decoding pH from QR code: {e}. Using dummy value.")
#         data["pH"] = 7  # Default to neutral pH

#     log_message(f"Final sensor data: {data}")
#     return data  # Ensure this is always executed

def calculate_motor_duration(severity, max_duration=30):
    """
    Calculate motor activation time based on severity.
    Scales the duration proportionally to the severity, capped at max_duration (default: 60 seconds)
    """
    base_duration = 1
    duration = min(base_duration + severity, max_duration)
    return duration

def activate_motor(motor_name, duration):
    """
    Activates a motor via the relay for a specified duration.
    Sends motor status (1 for active, 0 for inactive) to Wise-PaaS Datahub.
    """
    try:
        if motor_name not in relay_mapping:
            log_message(f"Motor {motor_name} not mapped to any relay.")
            return

        relay_channel = relay_mapping[motor_name]

        log_message(f"Activating {motor_name} motor via relay {relay_channel} for {duration} seconds.")
        activate_relay(relay_channel)
        upload_motor_data_to_datahub(motor_name, 1)

        time.sleep(duration)

        log_message(f"Deactivating {motor_name} motor via relay {relay_channel}.")
        deactivate_relay(relay_channel)
        upload_motor_data_to_datahub(motor_name, 0)

    except Exception as e:
        log_message(f"Error during motor activation/deactivation for {motor_name}: {str(e)}")

def control_solenoid(solenoid_name, action):
    """
    Controls solenoids by turning them ON or OFF.

    Parameters:
        solenoid_name (str): The name of the solenoid to control (e.g., "Sol1", "Sol2").
        action (str): Action to perform - "ON" to activate, "OFF" to deactivate.
    """
    try:
        if solenoid_name not in relay_mapping:
            log_message(f"Solenoid {solenoid_name} not mapped to any relay.")
            return

        relay_channel = relay_mapping[solenoid_name]

        if action.upper() == "ON":
            log_message(f"Turning ON {solenoid_name} via relay {relay_channel}.")
            activate_relay(relay_channel)
            upload_motor_data_to_datahub(solenoid_name, 1)
        elif action.upper() == "OFF":
            log_message(f"Turning OFF {solenoid_name} via relay {relay_channel}.")
            deactivate_relay(relay_channel)
            upload_motor_data_to_datahub(solenoid_name, 0)
        else:
            log_message(f"Invalid action '{action}' for {solenoid_name}. Use 'ON' or 'OFF'.")

    except Exception as e:
        log_message(f"Error controlling solenoid {solenoid_name}: {str(e)}")


def upload_motor_data_to_datahub(motor_name,status):
    """
    Upload data to Wise-PaaS Datahub.
    Includes sensor data, timestamp, watered status, and nutrient deficiency prediction.
    """
    try:
        edge_data = EdgeData()

        timestamp = datetime.now()
        edge_data.timestamp = timestamp

        edge_data.tagList.append(EdgeTag(deviceId="MotorStatus", tagName="StatusA", value= status if motor_name=="Solution A" else  0))
        edge_data.tagList.append(EdgeTag(deviceId="MotorStatus", tagName="StatusB", value= status if motor_name=="Solution B" else 0))
        edge_data.tagList.append(EdgeTag(deviceId="MotorStatus", tagName="StatusMix", value= status if motor_name=="Mixing Tank" else 0))
        edge_data.tagList.append(EdgeTag(deviceId="MotorStatus", tagName="StatusWater", value= status if motor_name=="Main Water" else 0))
        edge_data.tagList.append(EdgeTag(deviceId="MotorStatus", tagName="Sol1", value= status if motor_name=="Sol1" else 0))
        edge_data.tagList.append(EdgeTag(deviceId="MotorStatus", tagName="Sol2", value= status if motor_name=="Sol2" else 0))
        edge_data.tagList.append(EdgeTag(deviceId="MotorStatus", tagName="Sol3", value= status if motor_name=="Sol3" else 0))
        edge_data.tagList.append(EdgeTag(deviceId="MotorStatus", tagName="Sol4", value= status if motor_name=="Sol4" else 0))


        _edgeAgent.sendData(edge_data)

        log_message(
            f"Data uploaded to Wise-PaaS successfully. "
            f"Timestamp: {timestamp}, Motor_name: {motor_name}, "
            f"Status: {status}"
        )
        time.sleep(3)
        return True

    except Exception as e:
        log_message(f"Error uploading data to Wise-PaaS: {str(e)}")
        return False

def automated_correction(sensor_data):
    """
    Activate motors based on deficiencies in N, P, K.
    Use Solution A Motor for N and K, Solution B Motor for P.
    Always perform mixing if any deficiency is detected.
    """
    try:
        solution_a_needed = False
        solution_b_needed = False
        water_status = False

        log_message(f"Starting automated correction for sensor data: {sensor_data}")

        # Nitrogen and Potassium correction (Solution A)
        if sensor_data["N"] < THRESHOLDS["N"]:
            severity = THRESHOLDS["N"] - sensor_data["N"]
            log_message(f"Nitrogen deficiency detected: {sensor_data['N']} < {THRESHOLDS['N']} (Severity: {severity})")
            solution_a_needed = True

        if sensor_data["K"] < THRESHOLDS["K"]:
            severity = THRESHOLDS["K"] - sensor_data["K"]
            log_message(f"Potassium deficiency detected: {sensor_data['K']} < {THRESHOLDS['K']} (Severity: {severity})")
            solution_a_needed = True

        # Phosphorus correction (Solution B)
        if sensor_data["P"] < THRESHOLDS["P"]:
            severity = THRESHOLDS["P"] - sensor_data["P"]
            log_message(f"Phosphorus deficiency detected: {sensor_data['P']} < {THRESHOLDS['P']} (Severity: {severity})")
            solution_b_needed = True

        if solution_a_needed:
            duration = calculate_motor_duration(max(THRESHOLDS["N"] - sensor_data["N"], THRESHOLDS["K"] - sensor_data["K"]))
            activate_motor("Solution A", duration)

        if solution_b_needed:
            duration = calculate_motor_duration(THRESHOLDS["P"] - sensor_data["P"])
            activate_motor("Solution B", duration)

        if solution_a_needed or solution_b_needed:
            log_message("Deficiencies detected. Activating water motor for mixing.")
            control_solenoid("Sol4", "ON")
            time.sleep(OPT_TIME)
            activate_motor("Main Water", 10) # HERE: Must be same duration
            control_solenoid("Sol4", "OFF")
            time.sleep(OPT_TIME)
            control_solenoid("Sol5", "ON") # Added 1 more solenoid
            time.sleep(OPT_TIME)
            activate_motor("Mixing Tank", 10) #TODO: Change to 60s / suitable time
            control_solenoid("Sol5", "OFF")
            control_solenoid("Sol2", "ON")
            time.sleep(OPT_TIME)
            control_solenoid("Sol1", "ON")
            time.sleep(OPT_TIME)
            activate_motor("Mixing Tank", 10) # HERE: Must be same duration
            control_solenoid("Sol2", "OFF")
            control_solenoid("Sol1", "OFF")

        # Watering based on humidity or pH
        if (
            sensor_data["humidity"] < THRESHOLDS["humidity"] or
            sensor_data["pH"] > THRESHOLDS["pH"]
        ):
            severity_humidity = max(THRESHOLDS["humidity"] - sensor_data["humidity"], 0)
            severity_temp = max(sensor_data["pH"] - THRESHOLDS["pH"], 0)
            severity = max(severity_humidity, severity_temp)
            duration = calculate_motor_duration(severity)
            control_solenoid("Sol3", "ON")
            control_solenoid("Sol1", "ON")
            activate_motor("Main Water", duration)
            time.sleep(3)
            control_solenoid("Sol3", "OFF")
            control_solenoid("Sol1", "OFF")
            water_status = True

        log_message("Automated correction completed successfully.")
        return water_status
    except Exception as e:
        log_message(f"Error during correction: {str(e)}")
        return False

def check_buffer_folder():
    """
    Check for .png or .jpg files in the buffer folder.
    Run classification on each file, delete the file after processing, and record the output.
    """
    try:
        # Get list of files in the buffer folder
        all_files = os.listdir(BUFFER_FOLDER)
        log_message(f"All files in buffer: {all_files}")

        # Filter for .png and .jpg files
        files = [f for f in all_files if isinstance(f, str) and f.endswith((".png", ".jpg"))]
        if not files:
            log_message("No image files found in buffer.")
            return []

        log_message(f"Found image files: {files}")

        results = {}
        for file in files:
            file_path = os.path.join(BUFFER_FOLDER, file)

            # Run image classification
            classification_result = run_image_classification(file_path)

            # Log and store the result
            log_message(f"Classification result for {file}: {classification_result}")
            results[file] = classification_result

            # Delete the processed file
            os.remove(file_path)
            log_message(f"Deleted processed file: {file_path}")
        return results
    except Exception as e:
        log_message(f"Error checking buffer folder: {str(e)}")
        return {}

def run_image_classification(image_file):
    """
    Run image classification using the TFLite model and Edge TPU.
    test with k_2.png and n_1.png for accurate display
    """
    try:
        labels = read_label_file(LABEL_FILE)

        interpreter = make_interpreter(MODEL_FILE)
        interpreter.allocate_tensors()

        if common.input_details(interpreter, 'dtype') != np.uint8:
            raise ValueError("Only uint8 input type is supported.")

        size = common.input_size(interpreter)
        image = Image.open(image_file).convert('RGB').resize(size, Image.LANCZOS)

        params = common.input_details(interpreter, 'quantization_parameters')
        scale = params['scales']
        zero_point = params['zero_points']
        mean = 128.0
        std = 128.0

        if abs(scale * std - 1) < 1e-5 and abs(mean - zero_point) < 1e-5:
            common.set_input(interpreter, image)
        else:
            normalized_input = (np.asarray(image) - mean) / (std * scale) + zero_point
            np.clip(normalized_input, 0, 255, out=normalized_input)
            common.set_input(interpreter, normalized_input.astype(np.uint8))

        # Run inference
        log_message(f"Running inference on image: {image_file}")
        start_time = time.perf_counter()
        interpreter.invoke()
        inference_time = time.perf_counter() - start_time

        classes = classify.get_classes(interpreter, top_k=1)

        log_message(f"Inference time: {inference_time * 1000:.1f}ms")

        threshold = 0.5
        filtered_classes = [c for c in classes if c.score >= threshold]

        if filtered_classes:
            result = labels.get(filtered_classes[0].id, f"Unknown({filtered_classes[0].id})")
            confidence = filtered_classes[0].score
            return {"class": result, "confidence": confidence}
        else:
            return{"class": "No result", "confidence": 0.0}
    except Exception as e:
        log_message(f"Error during image classification: {str(e)}")
        return{"class": "Error", "confidence": 0.0}

# def convert_to_inverted_percentage(raw_value):
#     """Convert the raw register value to an inverted percentage (lower value = higher %)."""
#     if raw_value is None:
#         return None
#     # Invert the percentage by subtracting the scaled value from 100
#     inverted_percentage = 100 - ((raw_value / MAX_REGISTER_VALUE) * 100)
#     return inverted_percentage


def upload_data_to_wisepaas(sensor_data, prediction, watered_status,response):
    """
    Upload data to Wise-PaaS Datahub.
    Includes sensor data, timestamp, watered status, and nutrient deficiency prediction.
    """
    try:
        # Prepare data packet for Wise-PaaS
        edge_data = EdgeData()

        timestamp = datetime.now()
        edge_data.timestamp = timestamp

        # Add sensor data
        edge_data.tagList.append(EdgeTag(deviceId="LettuceSystem", tagName="Timestamp", value=timestamp.isoformat()))
        edge_data.tagList.append(EdgeTag(deviceId="LettuceSystem", tagName="N", value=sensor_data.get("N", 0)))
        edge_data.tagList.append(EdgeTag(deviceId="LettuceSystem", tagName="P", value=sensor_data.get("P", 0)))
        edge_data.tagList.append(EdgeTag(deviceId="LettuceSystem", tagName="K", value=sensor_data.get("K", 0)))
        edge_data.tagList.append(EdgeTag(deviceId="LettuceSystem", tagName="Humidity", value=sensor_data.get("humidity", 0)))
        edge_data.tagList.append(EdgeTag(deviceId="LettuceSystem", tagName="Response", value=response))
        edge_data.tagList.append(EdgeTag(deviceId="LettuceSystem", tagName="pH", value=sensor_data.get("pH", 0)))

        # Add watered status
        edge_data.tagList.append(EdgeTag(deviceId="LettuceSystem", tagName="Watered", value="Yes" if watered_status else "No"))

        # Add model prediction
        edge_data.tagList.append(EdgeTag(deviceId="LettuceSystem", tagName="Prediction", value=prediction.get("class", "Unknown")))
        edge_data.tagList.append(EdgeTag(deviceId="LettuceSystem", tagName="Confidence", value=prediction.get("confidence", 0.0)))

        # Send data to Wise-PaaS
        _edgeAgent.sendData(edge_data)

        # Log success
        log_message(
            f"Data uploaded to Wise-PaaS successfully. "
            f"Timestamp: {timestamp}, N: {sensor_data.get('N', 0)}, "
            f"P: {sensor_data.get('P', 0)}, K: {sensor_data.get('K', 0)}, "
            f"Humidity: {sensor_data.get('humidity', 0)}, Temp: {sensor_data.get('temperature', 0)}, "
            f"Water Status: {'Yes' if watered_status else 'No'}, "
            f"Prediction: {prediction.get('class', 'Unknown')}, Confidence: {prediction.get('confidence', 0.0)}"
        )
        return True

    except Exception as e:
        log_message(f"Error uploading data to Wise-PaaS: {str(e)}")

        # Save locally in case of failure
        backup_data = {
            "timestamp": timestamp.isoformat(),
            "sensor_data": sensor_data,
            "prediction": prediction,
            "watered_status": watered_status
        }
        with open("local_backup.json", "w") as backup:
            backup.write(str(backup_data))
        log_message("Data saved locally to local_backup.json due to upload failure.")
        return False


def main():
    """Main state machine loop."""
    try:
        setup_relays()

        _edgeAgent.on_connected = lambda agent, conn: log_message("Connected to Wise-PaaS.")
        _edgeAgent.on_disconnected = lambda agent, disc: log_message("Disconnected from Wise-PaaS.")
        _edgeAgent.connect()

        time.sleep(5)

        last_weather_fetch_date = None

        while True:
            current_time = datetime.now()

            if current_time.hour == 12 and last_weather_fetch_date != current_time.date():
                log_message("Fetching weather data at 12 PM.")
                forecast_response = fetch_forecast_data(API_KEY, LOCATION)
                if forecast_response:
                    forecast_data = process_forecast_data(forecast_response)
                    forecast_data = calculate_derivative(forecast_data)
                    watering_schedule = scheduled_watering_with_derivative(forecast_data)
                    watering_df = pd.DataFrame(watering_schedule)

                    plot_forecast(watering_df)
                    send_watering_schedule_to_wisepaas(watering_schedule)

                last_weather_fetch_date = current_time.date()
            else:
                log_message("Weather data fetch skipped; not 12 PM or already fetched for today.")

            sensor_data = fetch_sensor_data()
            response = analyze_crop_data(sensor_data)
            water_status = automated_correction(sensor_data)

            classification_results = check_buffer_folder()
            prediction = (
                classification_results[list(classification_results.keys())[0]]
                if classification_results
                else {"class": "None", "confidence": 0.0}
            )

            upload_data_to_wisepaas(sensor_data, prediction, water_status,response)

            log_message("Cycle complete. Waiting for the next iteration.")
            time.sleep(60)  # Run the loop every minute

    except KeyboardInterrupt:
        log_message("Program interrupted by user.")
    except Exception as e:
        log_message(f"Unexpected error: {str(e)}")
    finally:
        cleanup()
        log_message("GPIO cleanup complete. Program terminated.")




if __name__ == "__main__":
    main()
