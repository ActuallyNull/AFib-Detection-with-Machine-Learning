import wfdb
import neurokit2 as nk
import numpy as np
import pandas as pd

record = wfdb.rdrecord("mit-bih/100")
fs = record.fs
ecg_signal = record.p_signal[:, 0]
duration = 5
samples = fs * duration

_, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=fs)
plot = nk.events_plot(rpeaks['ECG_R_Peaks'][:5], ecg_signal[:6*fs])
plot.savefig("ecg.png")