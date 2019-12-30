import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


samples = pd.read_csv("Cancer_5_final.csv").values

X = samples[:-1, :-1].astype( 'float32' )
y = samples[:-1,11407]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(11408, activation='relu'),
  tf.keras.layers.Dropout(0.0),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
