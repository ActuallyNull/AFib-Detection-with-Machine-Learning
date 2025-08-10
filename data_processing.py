from sklearn.model_selection import train_test_split
import wfdb
import numpy as np
import os
import pandas as pd

dataset_folders = ["mit-bih", "training2017/training2017"]
records = []
labels = []
physionet_label_map = {'N': "Normal", 'A': "AFib", 'O': "Other Arrythmia"}

X = []
y = []

def load_physionet_dataset(physionet_path, reference_file_path):

    reference_df = pd.read_csv(reference_file_path, header=None, names=['record', 'label'])

    # loading the data
    for file in os.listdir(physionet_path):
        if file.endswith(".hea"):
            record_name = file[:-4]
            records.append(record_name)

    for name in records:
        nameLabel = reference_df.loc[reference_df['record'] == name, 'label'].values[0]
        if nameLabel == "N":
            labels.append("Normal")
        elif nameLabel == "A":
            labels.append("AFib")
        else:
            labels.append("Other Arrythmia")

    # entering it in dataset
    for paths in records:
        record = wfdb.rdrecord(physionet_path+"\\"+paths)
        ecg_signal = record.p_signal[:, 0]
        X.append(ecg_signal)

    for label in labels:
        y.append(label)

load_physionet_dataset("training2017\\training2017", "training2017\\training2017\\REFERENCE.csv")
