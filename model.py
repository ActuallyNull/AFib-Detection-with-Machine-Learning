import tensorflow as tf
from keras import layers, models
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from data_processing import create_train_val_test_splits
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
from keras import regularizers
import numpy as np

X_train, y_train, X_val, y_val, X_test, y_test, le = create_train_val_test_splits()

weights = compute_class_weight('balanced', classes=np.array([0,1,2]), y=y_train)
class_weights = dict(enumerate(weights))

callback = [
    EarlyStopping(monitor='val_loss', patience=7, verbose=1, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
]

def ecg_cnn(input_length, num_classes=3): # 0: afib, 1: normal, 2: other arrhythmia
    inputs = layers.Input(shape=(input_length, 1))

    x = layers.Conv1D(32, kernel_size=15, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(64, kernel_size=11, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(128, kernel_size=7, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(256, kernel_size=5, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

model = ecg_cnn(input_length=3000) # 300 hz for 10 sec = 3000 samples

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, 
          y_train, 
          epochs=100, 
          batch_size=64,
          callbacks=callback,
          class_weight=class_weights
          )

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_classes))

print("Classification Report:")
print(classification_report(y_test, y_pred_classes))

y_test_binary = label_binarize(y_test, classes=[0,1,2])

for i in range(y_pred.shape[1]):
    RocCurveDisplay.from_predictions(y_test_binary[:,i], y_pred[:,i], name=f"Class {i}")

plt.plot([0,1], [0,1], 'k--')
plt.xlabel("FPR (False Positive Rate)")
plt.ylabel("TPR (True Positive Rate)")
plt.title("OvR ROC Curve")
plt.legend()
plt.show()