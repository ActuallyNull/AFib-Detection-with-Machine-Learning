import tensorflow as tf
from keras import layers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from data_preperation import create_train_val_test_splits
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import label_binarize

X_train, y_train, X_val, y_val, X_test, y_test = create_train_val_test_splits()

callback = [
    EarlyStopping(monitor='val_loss', patience=7, verbose=1, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
]

model = tf.keras.Sequential([
    layers.Conv1D(60, 1),
    layers.Conv1D(50, 1),
    layers.Conv1D(40, 1),
    layers.GlobalAveragePooling1D(),
    layers.Dense(20, activation='relu'),
    layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, 
          y_train, 
          epochs=100, 
          batch_size=100,
          callbacks=callback,
          )

print(model.summary)

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