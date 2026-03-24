from focal_loss import SparseCategoricalFocalLoss
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import wfdb
import os
from scipy.signal import butter, filtfilt

# Load model with focal loss
model = keras.models.load_model(
    "model.keras",
    custom_objects={"SparseCategoricalFocalLoss": SparseCategoricalFocalLoss}
)

# --- Configuration ---
CLASS_NAMES = ["Normal", "AFib", "Other"]  # <-- update with your actual class labels
INPUT_LENGTH = 3000  # <-- change to the length your model expects

def butter_bandpass_filter(signal, fs, low_pass=0.5, high_pass=50, order=5):
    nyq = 0.5 * fs
    low = low_pass / nyq
    high = high_pass / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def load_ecg(file_path: str) -> tuple[np.ndarray, int]:
    """
    Load ECG data from either CSV or WFDB format.
    Returns a 1D numpy array of the ECG signal and the sampling frequency.
    """
    ext = os.path.splitext(file_path)[-1].lower()
    fs = 300 # default sampling frequency

    if ext == ".csv":
        df = pd.read_csv(file_path, header=None)
        signal = df.values.flatten()

    elif ext == ".dat" or ext == ".hea":
        # Strip extension for wfdb
        record_name = file_path.replace(ext, "")
        record = wfdb.rdrecord(record_name)
        # Use the first channel by default
        signal = record.p_signal[:, 0]
        fs = record.fs

    else:
        raise ValueError(f"Unsupported file format: {ext}")

    return signal, fs

def preprocess_ecg(signal: np.ndarray, fs: int) -> np.ndarray:
    """
    Preprocess ECG signal into model-ready input.
    Pads/truncates, normalizes, reshapes.
    """
    # 1. Filter the signal
    signal = butter_bandpass_filter(signal, fs)

    # 2. Pad or truncate
    if len(signal) < INPUT_LENGTH:
        signal = np.pad(signal, (0, INPUT_LENGTH - len(signal)))
    else:
        signal = signal[:INPUT_LENGTH]

    # 3. Normalize (min-max)
    signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-8)

    # 4. Add batch + channel dims → (1, length, 1)
    signal = np.expand_dims(signal, axis=(0, -1))
    return signal.astype(np.float32)

def classify_ecg(file_path: str):
    """
    Run model prediction on an ECG file (CSV or WFDB).
    """
    raw_signal, fs = load_ecg(file_path)
    x = preprocess_ecg(raw_signal, fs)

    preds = model.predict(x)
    pred_class = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(tf.nn.softmax(preds, axis=1)))

    return {
        "predicted_class": CLASS_NAMES[pred_class],
        "confidence": confidence,
        "raw_logits": preds.tolist()
    }

base_file_path = "training2017/training2017/"

if __name__ == "__main__":

    while True:
        test_file = input("Enter the path to the ECG file: ")
        result = classify_ecg(base_file_path + test_file)
        print("Prediction:", result)