# eeg_controller.py
# Runtime bridge: reads EEG feature JSON lines from an EEG bridge or logger (stdin or serial),
# predicts command with saved model, and sends single-letter commands to Arduino.
#
# Expected incoming lines (JSON per line), e.g.:
# {"attention": 72, "meditation": 30, "blink": 0, "raw": 12345}
#
# Label mapping example used in training: labels = {"forward","back","left","right","stop"}

import serial
import time
import json
import joblib
import numpy as np
from collections import deque
import argparse
import sys

parser = argparse.ArgumentParser(description="EEG -> Arduino controller")
parser.add_argument("--eeg-source", default=None,
                    help="Serial port where EEG bridge sends JSON lines (e.g. COM5 or /dev/ttyUSB0). If omitted, reads stdin.")
parser.add_argument("--baud", type=int, default=115200, help="baud rate for both ports")
parser.add_argument("--arduino-port", required=True, help="Serial port for Arduino (e.g. COM3 or /dev/ttyACM0)")
parser.add_argument("--model", default="models/eeg_model.joblib", help="path to trained model file")
parser.add_argument("--window", type=int, default=5, help="sliding window size for smoothing")
args = parser.parse_args()

# Load model
model = joblib.load(args.model)
print("Loaded model:", args.model)

# Open Arduino serial
arduino = serial.Serial(args.arduino_port, args.baud, timeout=1)
time.sleep(2)  # allow Arduino to reset

# Setup EEG source
if args.eeg_source:
    eeg_ser = serial.Serial(args.eeg_source, args.baud, timeout=1)
    read_source = eeg_ser
    print("Reading EEG from serial:", args.eeg_source)
else:
    read_source = sys.stdin
    print("Reading EEG from stdin (pipe JSON lines)")

# Simple smoothing buffers for numeric features
window = args.window
buffers = {}
feature_order = None  # will set after first input
bufq = {}

def send_command(cmd_letter):
    """Send single-character command to Arduino (F,B,L,R,S)"""
    if not cmd_letter:
        return
    if isinstance(cmd_letter, str):
        cmd = cmd_letter.strip()[0].upper()
    else:
        cmd = str(cmd_letter)
    arduino.write(cmd.encode('ascii'))
    print(f"-> Arduino: {cmd}")

def parse_line(line):
    try:
        obj = json.loads(line)
        return obj
    except Exception as e:
        return None

def extract_features(obj):
    # Example: use attention, meditation, blink if present
    features = []
    keys = []
    for k in ["attention", "meditation", "blink", "raw"]:
        if k in obj:
            features.append(float(obj[k]))
            keys.append(k)
    # fallback: use all numeric fields
    if not features:
        for k,v in obj.items():
            if isinstance(v, (int, float)):
                features.append(float(v))
                keys.append(k)
    return keys, np.array(features, dtype=float)

try:
    while True:
        line = read_source.readline()
        if not line:
            time.sleep(0.01)
            continue
        if isinstance(line, bytes):
            line = line.decode('utf-8', errors='ignore')
        line = line.strip()
        if not line:
            continue

        obj = parse_line(line)
        if obj is None:
            print("Ignored line (not JSON):", line[:100])
            continue

        keys, feat = extract_features(obj)
        if feature_order is None:
            feature_order = keys
            print("Feature order set to:", feature_order)
            # init buffers
            for k in feature_order:
                bufq[k] = deque(maxlen=window)

        # align with feature_order
        sample = []
        for k in feature_order:
            val = obj.get(k, 0.0)
            bufq[k].append(float(val))
            sample.append(float(np.mean(bufq[k])))

        sample = np.array(sample).reshape(1, -1)
        # If model expects a particular number of features, ensure shape matches
        if sample.shape[1] != model.n_features_in_:
            print("Feature size mismatch:", sample.shape, "model expects", model.n_features_in_)
            continue

        pred = model.predict(sample)[0]  # expected labels like: forward/back/left/right/stop
        prob = None
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(sample).max()
        print(f"Predicted: {pred}", f" (p={prob:.2f})" if prob is not None else "")

        # Map label to single-letter command for Arduino
        mapping = {
            "forward": "F",
            "back": "B",
            "left": "L",
            "right": "R",
            "stop": "S",
            "neutral": "S"
        }
        cmd = mapping.get(pred, "S")
        send_command(cmd)

except KeyboardInterrupt:
    print("Exiting...")
finally:
    try:
        arduino.close()
    except:
        pass
    if args.eeg_source:
        try:
            eeg_ser.close()
        except:
            pass
