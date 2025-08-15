import wfdb
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import butter, filtfilt
from imblearn.over_sampling import SMOTE
from data_augmentor import ECG_Augmentor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import gc

physionet_label_map = {'N': "Normal", 'A': "AFib", 'O': "Other Arrythmia"}
duration = 10 # seconds

def butter_bandpass_filter(signal, fs, low_pass=0.5, high_pass=50, order=5):
    nyq = 0.5 * fs
    low = low_pass / nyq
    high = high_pass / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def smote_resampling(X, y):
    # Only apply SMOTE if there are at least two classes
    if len(np.unique(y)) > 1:
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled
    else:
        return X, y

def standardise_dataset(X):
    X_scaled = np.array([(x - np.mean(x)) / np.std(x) for x in X])
    return X_scaled

def normalise_dataset(X):
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)
    return X_norm

def load_physionet_dataset(physionet_path, reference_file_path, X, y):
    reference_df = pd.read_csv(reference_file_path, header=None, names=['record', 'label'])
    records = []
    labels = []

    # Only keep records that are not noisy
    for file in os.listdir(physionet_path):
        if file.endswith(".hea"):
            record_name = file[:-4]
            label_row = reference_df.loc[reference_df['record'] == record_name]
            if label_row.empty:
                continue  # Skip if not found in reference
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
    X, y = load_physionet_dataset(physionet_path, reference_file_path, X, y)
    X_filtered = np.array([butter_bandpass_filter(x, fs=fs) for x in X])
    # Do not normalise or augment here!
    y_processed = y
    X_balanced, y_balanced = smote_resampling(X_filtered, y_processed)
    return X_balanced, y_balanced


def create_train_val_test_splits(fs=300):
    X = []
    y = []
    # Use strong augmentations for training data
    augmentor = ECG_Augmentor(fs=fs, strong=True)
    # Load and preprocess only, avoid unnecessary copies
    X_processed, y_processed = preprocess_dataset(
        "training2017/training2017",
        "training2017/training2017/REFERENCE.csv",
        X, y
    )
    # Use float32 to save memory
    X_processed = X_processed.astype(np.float32)
    # Reshape for CNN
    X_cnn = np.expand_dims(X_processed, axis=-1)
    le = LabelEncoder()
    y_cnn = le.fit_transform(y_processed)
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_cnn, y_cnn, test_size=0.3, random_state=42, stratify=y_cnn
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # normal augmentation
    batch_size = 500  # Adjust as needed for your RAM
    X_train_aug = []
    for i in range(0, len(X_train), batch_size):
        batch = X_train[i:i+batch_size]
        batch_aug = []
        for x in batch:
            if augmentor.strong and np.random.rand() < 0.2 and len(batch) > 1:
                # Mixup with another random signal in the batch
                idx = np.random.choice([j for j in range(len(batch)) if not np.allclose(batch[j], x)])
                x_aug = augmentor.augment(x.squeeze(), mix_signal=batch[idx].squeeze())
            else:
                x_aug = augmentor.augment(x.squeeze())
            batch_aug.append(x_aug)
        batch_aug = np.expand_dims(np.array(batch_aug, dtype=np.float32), axis=-1)
        X_train_aug.append(batch_aug)
    X_train_aug = np.concatenate(X_train_aug, axis=0)

    # extra augmentation for class 2 cause it has more trouble
    other_class_idx = le.transform(["Other Arrythmia"])[0]
    other_mask = (y_train == other_class_idx)
    X_other = X_train[other_mask]
    y_other = y_train[other_mask]

    extra_augmented = []
    for x in X_other:
        for _ in range(3):  # triple augmentations for more variability
            extra_augmented.append(augmentor.augment(x.squeeze()))
    extra_augmented = np.expand_dims(np.array(extra_augmented, dtype=np.float32), axis=-1)
    y_other_extra = np.full(len(extra_augmented), other_class_idx)

    # Append extra augmented "Other Arrhythmia" samples
    X_train_aug = np.concatenate([X_train_aug, extra_augmented], axis=0)
    y_train = np.concatenate([y_train, y_other_extra], axis=0)

    # Clean up memory
    del X, y, X_processed, X_cnn, X_train, X_temp, batch, batch_aug
    gc.collect()

    return X_train_aug, y_train, X_val, y_val, X_test, y_test, le