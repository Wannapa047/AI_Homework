import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from keras.api.models import Sequential
from keras.api.layers import Dense, SimpleRNN

# ตั้งค่าพารามิเตอร์
N = 100
step = 11 # Step Size
n_train = int(N * 0.7)  # 70% สำหรับการ Train

# สร้างสัญญาณใหม่ (Sine wave + Noise)
t = np.arange(N)
y = np.sin(0.05 * t * 10) * 20 + 10 + np.random.randn(N) * 5  # ปรับสเกลของสัญญาณ

# ฟังก์ชันสำหรับแปลงข้อมูลให้เป็น Matrix
def convert_to_matrix(data, step):
    X, Y = [], []
    for i in range(len(data) - step):
        X.append(data[i:i + step])
        Y.append(data[i + step])
    return np.array(X), np.array(Y)

# แบ่งข้อมูลเป็นชุด Train และ Test
train, test = y[:n_train], y[n_train:]
x_train, y_train = convert_to_matrix(train, step)
x_test, y_test = convert_to_matrix(test, step)

# Reshape ข้อมูลสำหรับ RNN
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

# สร้างโมเดล RNN
model = Sequential()
model.add(SimpleRNN(32, input_shape=(step, 1), activation="relu")) 
model.add(Dense(1))

# Compile โมเดล
model.compile(optimizer="adam", loss="mse")

# Train โมเดล
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1)

# พยากรณ์แบบต่อเนื่อง (Recursive Prediction)
predicted = []
input_sequence = y[:step]  # เริ่มต้นด้วยค่าจริงในช่วงแรก
for _ in range(len(y) - step):
    # Reshape ข้อมูลก่อนใส่โมเดล
    input_sequence_reshaped = input_sequence[-step:].reshape(1, step, 1)
    # พยากรณ์ค่าใหม่
    next_value = model.predict(input_sequence_reshaped, verbose=0)
    # เพิ่มค่าที่พยากรณ์ลงในผลลัพธ์
    predicted.append(next_value[0, 0])
    # อัปเดตลำดับข้อมูลสำหรับการพยากรณ์ครั้งถัดไป
    input_sequence = np.append(input_sequence, next_value)

# ฟังก์ชันสำหรับ Plot การเปรียบเทียบ
def plot_comparison(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(y_true)), y_true, label="Original", color='blue', linestyle='-')
    plt.plot(np.arange(len(y_pred)), y_pred, label="Predict", color='red', linestyle='--')
    plt.axvline(x=70, color='magenta', linestyle='-', linewidth=2)  # เส้นแบ่งที่ตำแหน่ง 70
    plt.ylim(-22, 39)  # ปรับสเกลแกน Y
    plt.xticks(np.arange(0, 101, 20))  # กำหนดแกน x เป็น [0, 20, 40, 60, 80, 100]
    plt.title("Comparison of Original and Predicted Data")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

# เรียกใช้ฟังก์ชัน Plot
plot_comparison(y, predicted)