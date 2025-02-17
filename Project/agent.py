import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# สร้างโมเดล Q-Network
def build_model(state_size, action_size):
    model = tf.keras.Sequential()
    model.add(layers.InputLayer(input_shape=(state_size,)))
    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dense(action_size, activation='linear'))  # output layer with linear activation for Q-values
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

# การทดสอบโมเดล
state_size = 5  # ตัวอย่าง: ขนาดของ state space
action_size = 3  # ตัวอย่าง: จำนวน actions ที่มี
model = build_model(state_size, action_size)
model.summary()  # แสดงรายละเอียดของโมเดล
