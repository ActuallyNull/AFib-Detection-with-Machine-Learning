import tensorflow as tf
from keras import layers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from data_preperation import create_train_val_test_splits
from sklearn.metrics import confusion_matrix, classification_report

X_train, y_train, X_val, y_val, X_test, y_test = create_train_val_test_splits()

callback = [
    EarlyStopping(monitor='val_loss', patience=7, verbose=1, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
]


model = tf.keras.Sequential([
    layers.Conv1D(32, kernel_size=5),
    layers.Conv1D(32, kernel_size=5),
    layers.MaxPool1D(pool_size=2),
    layers.Conv1D(64, kernel_size=5),
    layers.Conv1D(64, kernel_size=5),
    layers.MaxPool1D(pool_size=2),
    layers.Conv1D(128, kernel_size=5),
    layers.Conv1D(128, kernel_size=5),
    layers.Conv1D(128, kernel_size=5),
    layers.MaxPool1D(pool_size=2),
    layers.Conv1D(256, kernel_size=5),        
    layers.Conv1D(256, kernel_size=5),        
    layers.Conv1D(256, kernel_size=5),        
    layers.MaxPool1D(pool_size=2),
    layers.Conv1D(256, kernel_size=5),        
    layers.Conv1D(256, kernel_size=5),        
    layers.Conv1D(256, kernel_size=5),        
    layers.MaxPool1D(pool_size=2),
    layers.GlobalAveragePooling1D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, 
          y_train, 
          epochs=100, 
          batch_size=64,
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