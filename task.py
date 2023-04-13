import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Fix the random initialization of model weights
np.random.seed(42)
tf.random.set_seed(42)

# Load Iris data
data = load_iris()
X, y = data['data'], data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the model
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(10, activation='relu', input_shape=(4,)),
  tf.keras.layers.Dense(3, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model on the training set
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Save the model
model.save('iris_model.h5')

# Load the model
loaded_model = tf.keras.models.load_model('iris_model.h5')

# Create new data for prediction
new_data = np.array([[5.1, 3.5, 1.4, 0.2], [6.2, 2.9, 4.3, 1.3], [7.3, 2.8, 6.3, 1.8]])

# Normalize the new data
new_data = scaler.transform(new_data)

# Make predictions on the new data
predictions = loaded_model.predict(new_data)
print(predictions)

# Create accuracy and loss plots
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model performance')
plt.ylabel('Accuracy / Loss')
plt.xlabel('Epoch')
plt.legend(['Train accuracy', 'Test accuracy', 'Train loss', 'Test loss'], loc='upper left')
plt.show()
