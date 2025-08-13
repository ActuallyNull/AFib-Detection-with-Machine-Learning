import wfdb
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import butter, filtfilt, resample
from imblearn.over_sampling import SMOTE
from data_augmentor import ECG_Augmentor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import gc
from ecg_augmentor import ECG_Augmentor

# ====================
# Signal Processing
# ====================
def butter_bandpass_filter(signal, fs, low_pass=0.5, high_pass=50, order=5):
    nyq = 0.5 * fs
    low = low_pass / nyq
    high = high_pass / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def load_physionet_dataset(physionet_path, reference_file_path, duration=10):
    physionet_label_map = {'N': "Normal", 'A': "AFib", 'O': "Other Arrythmia"}
    reference_df = pd.read_csv(reference_file_path, header=None, names=['record', 'label'])

    X, y = [], []
    for file in os.listdir(physionet_path):
        if file.endswith(".hea"):
            record_name = file[:-4]
            label_row = reference_df.loc[reference_df['record'] == record_name]
            if label_row.empty:
                continue
            label_code = label_row['label'].values[0]
            if label_code == "~":
                continue  # Skip noisy signals
            records.append(record_name)
            labels.append(physionet_label_map.get(label_code, "Other Arrythmia"))

    # Load and pad signals
    for record_name in records:
        record = wfdb.rdrecord(os.path.join(physionet_path, record_name))
        fs = record.fs
        samples = fs * duration
        ecg_signal = record.p_signal[:, 0]
        trunc_ecg = ecg_signal[:samples]
        if len(trunc_ecg) < samples:
            trunc_ecg = np.pad(trunc_ecg, (0, samples - len(trunc_ecg)), 'constant')
        X.append(trunc_ecg)

    y.extend(labels)
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    return X, y

def preprocess_dataset(physionet_path, reference_file_path, X, y, fs=300):
    augmentor = ECG_Augmentor(fs=fs)
    X, y = load_physionet_dataset(physionet_path, reference_file_path, X, y)
    X_filtered = np.array([butter_bandpass_filter(x, fs=fs) for x in X])
    X_augmented = np.array([augmentor.augment(x) for x in X_filtered])
    X_processed = normalise_dataset(X_augmented) # For DCNN, normalization is preferred
    y_processed = y
    X_balanced, y_balanced = smote_resampling(X_processed, y_processed)
    return X_balanced, y_balanced
