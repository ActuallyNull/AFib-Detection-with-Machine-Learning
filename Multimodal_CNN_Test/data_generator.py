import numpy as np
from keras.utils import Sequence

class ECGDataGenerator(Sequence):
    def __init__(self, X, y, features, batch_size=64, augmentor=None, shuffle=True):
        self.X = X
        self.y = y
        self.features = features
        self.batch_size = batch_size
        self.augmentor = augmentor
        self.shuffle = shuffle
        self.indices = np.arange(len(self.X))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        idx = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        X_batch = self.X[idx].copy()
        y_batch = self.y[idx]
        f_batch = self.features[idx]

        if self.augmentor:
            X_batch = np.array([self.augmentor.augment(x.squeeze()) for x in X_batch])
        else:
            X_batch = np.array([x.squeeze() for x in X_batch])

        # Standardize per-sample (preserves morphology)
        X_batch = (X_batch - X_batch.mean(axis=1, keepdims=True)) / \
                  (X_batch.std(axis=1, keepdims=True) + 1e-8)

        X_batch = np.expand_dims(X_batch, -1)
        return [X_batch, f_batch], y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
