import neurokit2 as nk
import wfdb
import numpy as np

fs = 500
duration = 10  # seconds

def save_ecg(ecg_signal, filename):
    ecg_signal = ecg_signal.reshape(-1, 1)
    wfdb.wrsamp(record_name=filename,
                fs=fs,
                units=['mV'],
                sig_name=['ECG'],
                p_signal=ecg_signal)
    print(f"Saved {filename}.dat and {filename}.hea")

# 1. Normal sinus rhythm (~60 bpm)
ecg_normal = nk.ecg_simulate(duration=duration, sampling_rate=fs, heart_rate=60)
save_ecg(ecg_normal, 'ecg_normal')

# 2. Tachycardia (~120 bpm)
ecg_tachy = nk.ecg_simulate(duration=duration, sampling_rate=fs, heart_rate=120)
save_ecg(ecg_tachy, 'ecg_tachycardia')

# 3. Atrial fibrillation (irregular heart rate)
# Introduce heart rate variability and irregular RR intervals
ecg_afib = nk.ecg_simulate(duration=duration, sampling_rate=fs, heart_rate=80, method="simple", noise=0.02)
# Add some irregularity to simulate AFib
rr_intervals = nk.ecg_simulate(duration=duration, sampling_rate=fs, heart_rate=80, method="ecgsyn")
rr_irregular = rr_intervals + np.random.normal(0, 0.1, size=rr_intervals.shape)
save_ecg(ecg_afib, 'ecg_afib')
