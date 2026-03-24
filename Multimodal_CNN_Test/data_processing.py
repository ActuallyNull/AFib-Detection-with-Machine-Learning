import wfdb
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt
from data_augmentor import ECG_Augmentor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import neurokit2 as nk
from sklearn.cluster import KMeans
import gc

physionet_label_map = {'N': "Normal", 'A': "AFib", 'O': "Other Arrythmia"}
duration = 10

def butter_bandpass_filter(signal, fs, low=0.5, high=50, order=5):
    nyq = fs * 0.5
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, signal)

def extract_features(signals, fs):
    feats = []
    for sig in signals:
        try:
            proc, info = nk.ecg_process(sig, sampling_rate=fs)
            f = nk.ecg_intervalrelated(proc, sampling_rate=fs)
        except:
            f = pd.DataFrame([np.zeros(10)])  # fallback
        feats.append(f)
    df = pd.concat(feats, ignore_index=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(), inplace=True)
    df.fillna(0, inplace=True)
    return df

def load_physionet(path, ref_path):
    ref = pd.read_csv(ref_path, header=None, names=['record','label'])
    X, y = [], []

    for file in os.listdir(path):
        if file.endswith(".hea"):
            rec = file[:-4]
            row = ref[ref.record == rec]
            if row.empty: continue
            label = row.label.values[0]
            if label == "~": continue

            record = wfdb.rdrecord(os.path.join(path, rec))
            fs = record.fs
            sig = record.p_signal[:,0]
            samples = fs * duration
            sig = sig[:samples]
            if len(sig) < samples:
                sig = np.pad(sig, (0, samples-len(sig)))

            X.append(sig)
            y.append(physionet_label_map.get(label, "Other Arrythmia"))

    return np.array(X, np.float32), np.array(y), fs

def create_train_val_test_splits():
    X, y, fs = load_physionet("training2017/training2017",
                              "training2017/training2017/REFERENCE.csv")

    X = np.array([butter_bandpass_filter(x, fs) for x in X], np.float32)

    features = extract_features(X, fs)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    base_names = list(le.classes_)

    X_train, X_temp, y_train, y_temp, f_train, f_temp = train_test_split(
        X, y_enc, features, test_size=0.3, random_state=42, stratify=y_enc)

    X_val, X_test, y_val, y_test, f_val, f_test = train_test_split(
        X_temp, y_temp, f_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # Subclassing
    other_label = "Other Arrythmia"
    other_idx = le.transform([other_label])[0]
    mask = (y_train == other_idx)

    if mask.any():
        f_other = f_train[mask]
        subs = KMeans(n_clusters=3, random_state=42).fit_predict(f_other)
        unique = np.unique(subs)

        new_names = [f"{other_label}_{i}" for i in unique]
        class_names = base_names + new_names

        y_new = y_train.copy()
        start = len(base_names)
        mapping = {s: start+i for i,s in enumerate(unique)}
        y_new[mask] = np.array([mapping[s] for s in subs])
        y_train = y_new
    else:
        class_names = base_names

    scaler = StandardScaler()
    f_train = scaler.fit_transform(f_train)
    f_val = scaler.transform(f_val)
    f_test = scaler.transform(f_test)

    return X_train, y_train, X_val, y_val, X_test, y_test, f_train, f_val, f_test, class_names, scaler
