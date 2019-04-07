import tensorflow as tf
from tensorflow import keras as k
from tensorflow.keras import layers
import numpy as np

print("tensorflow version: ", tf.VERSION)
print("keras version:      ", tf.keras.__version__)

print("Build model...")
model = tf.keras.Sequential([
    layers.Dense(5, activation='tanh'),
    layers.Dense(10, activation='softplus'),
    layers.Dense(10, activation='softmax')
])

print("Compile model...")
model.compile(
    optimizer=tf.train.AdamOptimizer(0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


print("Build data...")
def answer(value):
    res = np.zeros(10)
    res[value // 2] = 1
    return res

def convert(arr):
    return np.array([answer(x) for x in arr])

def ex(t):
    return np.expand_dims(t, 1)

data = np.random.randint(0, 20, 100000)
labels = convert(data)

val_data = np.random.randint(0, 20, 100)
val_labels = convert(val_data)

print(model.to_json())

model.fit(
    ex(data.astype(float)), labels,
    epochs=20, steps_per_epoch=100,
    batch_size=32,
    validation_data=(ex(val_data.astype(float)), val_labels)
)

res=model.predict(ex([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
print(res)
print(np.argmax(res, axis=1))
