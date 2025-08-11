import tensorflow as tf
from keras import layers
from data_preperation import create_train_val_test_splits
from sklearn.metrics import confusion_matrix, classification_report

X_train, y_train, X_val, y_val, X_test, y_test = create_train_val_test_splits()

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

model.fit(X_train, y_train, epochs=10, batch_size=32)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_classes))

print("Classification Report:")
print(classification_report(y_test, y_pred_classes))