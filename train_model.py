import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

X = np.random.rand(100,128,128,3)
y = np.random.randint(0,2,100)

model = models.Sequential([
    layers.Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dense(1,activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X, y, epochs=3)

model.save("model.h5")

print("Model saved!")