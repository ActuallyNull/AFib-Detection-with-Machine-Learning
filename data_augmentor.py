import numpy as np
from scipy.signal import resample

class ECG_Augmentor:
    def __init__(self, fs=300):
        self.fs = fs

    def add_gaussian_noise(self, signal, noise_level=0.01):
        noise = np.random.normal(0, noise_level * np.max(signal), size=signal.shape)
        return signal + noise

    def baseline_wander(self, signal, freq=0.3, amp=0.05):
        t = np.arange(len(signal)) / self.fs
        drift = amp * np.sin(2 * np.pi * freq * t)
        return signal + drift

    def time_shift(self, signal, shift_max=0.05):
        shift = int(np.random.uniform(-shift_max, shift_max) * len(signal))
        return np.roll(signal, shift)

    def time_scale(self, signal, scale_range=(0.95, 1.05)):
        scale = np.random.uniform(*scale_range)
        new_len = int(len(signal) * scale)
        scaled = resample(signal, new_len)
        if len(scaled) > len(signal):
            return scaled[:len(signal)]
        else:
            return np.pad(scaled, (0, len(signal) - len(scaled)), mode='constant')

    def amplitude_scale(self, signal, scale_range=(0.95, 1.05)):
        scale = np.random.uniform(*scale_range)
        return signal * scale

    def amplitude_shift(self, signal, shift_range=(-0.05, 0.05)):
        shift = np.random.uniform(*shift_range) * np.max(signal)
        return signal + shift

    def mixup(self, sig1, sig2, alpha=0.2):
        lam = np.random.beta(alpha, alpha)
        return lam * sig1 + (1 - lam) * sig2, lam

    def augment(self, signal, mix_signal=None):
        """Apply a random set of augmentations with tuned probabilities for AFib detection"""
        aug_signal = signal.copy()

        if np.random.rand() < 0.5:
            aug_signal = self.add_gaussian_noise(aug_signal)
        if np.random.rand() < 0.4:
            aug_signal = self.baseline_wander(aug_signal)
        if np.random.rand() < 0.3:
            aug_signal = self.time_shift(aug_signal)
        if np.random.rand() < 0.2:
            aug_signal = self.time_scale(aug_signal)
        if np.random.rand() < 0.3:
            aug_signal = self.amplitude_scale(aug_signal)
        if np.random.rand() < 0.2:
            aug_signal = self.amplitude_shift(aug_signal)
        if mix_signal is not None and np.random.rand() < 0.2:
            aug_signal, _ = self.mixup(aug_signal, mix_signal)

        return aug_signal