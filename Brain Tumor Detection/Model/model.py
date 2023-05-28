import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

DATA_DIR_YES = r"C:\Users\Aryan\FLASK\Brain Tumor Detection\brain_tumor_dataset\yes"
DATA_DIR_NO = r"C:\Users\Aryan\FLASK\Brain Tumor Detection\brain_tumor_dataset\no"

encoder = OneHotEncoder()
encoder.fit([[0], [1]])

# 0 - Tumor
# 1 - Normal

# This cell updates result list for images with tumor

data = []
paths = []
result = []

for r, d, f in os.walk(DATA_DIR_YES):
    for file in f:
        if '.jpg' in file:
            paths.append(os.path.join(r, file))

for path in paths:
    img = Image.open(path)
    img = img.resize((128, 128))
    img = np.array(img)
    if (img.shape == (128, 128, 3)):
        data.append(np.array(img))
        result.append(encoder.transform([[0]]).toarray())

# This cell updates result list for images without tumor

paths = []
for r, d, f in os.walk(DATA_DIR_NO):
    for file in f:
        if '.jpg' in file:
            paths.append(os.path.join(r, file))

for path in paths:
    img = Image.open(path)
    img = img.resize((128, 128))
    img = np.array(img)
    if (img.shape == (128, 128, 3)):
        data.append(np.array(img))
        result.append(encoder.transform([[1]]).toarray())

data = np.array(data)

result = np.array(result)
result = result.reshape(134, 2)

x_train, x_val, y_train, y_val = train_test_split(
    data, result, test_size=0.2, shuffle=True, random_state=0)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=64,
                           kernel_size=3,
                           activation="relu",
                           input_shape=(128, 128, 3)),
    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=2,
                              padding="valid"),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(2, activation="softmax")
])

model.compile(loss="binary_crossentropy",
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"]
              )


model_history = model.fit(x_train, y_train,
                          epochs=32, batch_size=36,
                          validation_data=(x_val, y_val))

model.save('brain_tumor_detection_model.h5')
model.summary()

plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Accuracy', 'Validation Accuracy'], loc='upper right')
plt.show()

plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Loss', 'Validation Loss'], loc='upper right')
plt.show()
