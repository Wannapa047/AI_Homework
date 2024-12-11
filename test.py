import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# 1. สร้างข้อมูลสำหรับ Class A และ Class B
x_a, y_a = make_blobs(n_samples=100,  # จำนวนตัวอย่างข้อมูลใน Class A
                      n_features=2,   # จำนวน Features = 2 (2 มิติ)
                      centers=[(2.0, 2.0)],  # ศูนย์กลางของ Class A
                      cluster_std=0.75,  # ส่วนเบี่ยงเบนมาตรฐาน (spread ของข้อมูล)
                      random_state=42)  # ค่า seed เพื่อให้ได้ผลลัพธ์เดิมทุกครั้ง

x_b, y_b = make_blobs(n_samples=100,  # จำนวนตัวอย่างข้อมูลใน Class B
                      n_features=2,   # จำนวน Features = 2 (2 มิติ)
                      centers=[(3.0, 3.0)],  # ศูนย์กลางของ Class B
                      cluster_std=0.75,  # ส่วนเบี่ยงเบนมาตรฐาน (spread ของข้อมูล)
                      random_state=42)  # ค่า seed เดียวกันสำหรับ reproducibility

# 2. รวมข้อมูลจาก Class A และ B เข้าด้วยกัน
X = np.vstack((x_a, x_b))  # รวมข้อมูล Class A และ B
y = np.hstack((np.zeros(100), np.ones(100)))  # สร้าง Labels (0 = Class A, 1 = Class B)

# 3. ปรับขนาดข้อมูลด้วย StandardScaler
scaler = StandardScaler()  # นิยาม Scaler
X_scaled = scaler.fit_transform(X)  # ปรับขนาดข้อมูลให้อยู่ในช่วงที่เหมาะสม (ค่าเฉลี่ย=0, ส่วนเบี่ยงเบนมาตรฐาน=1)

# 4. แบ่งข้อมูลออกเป็นชุดเทรนและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42)  # แบ่งข้อมูล 70% สำหรับเทรน, 30% สำหรับทดสอบ

# 5. สร้างและเทรน Neural Network
mlp = MLPClassifier(hidden_layer_sizes=(10, 10),  # Neural Network มี 2 hidden layers, แต่ละ layer มี 10 neurons
                    max_iter=1000,  # จำนวนรอบในการเทรนสูงสุด
                    random_state=42)  # ค่า seed สำหรับ reproducibility
mlp.fit(X_train, y_train)  # เทรนโมเดลด้วยชุดข้อมูลเทรน

# 6. สร้าง Grid สำหรับการวาด Decision Boundary
x1_range = np.linspace(3, -3, 500)  # สร้างช่วงค่าของ Feature x1
x2_range = np.linspace(3, -3, 500)  # สร้างช่วงค่าของ Feature x2
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)  # สร้าง grid (ตาราง) ที่ใช้สำหรับการทำนาย

# 7. ทำนายผลลัพธ์บน Grid
grid_points = np.c_[x1_grid.ravel(), x2_grid.ravel()]  # แปลง grid เป็น array 2D สำหรับการทำนาย
predictions = mlp.predict(grid_points).reshape(x1_grid.shape)  # ทำนายและ reshape ให้มีขนาดเหมือน grid

# 8. พล็อต Decision Boundary และข้อมูล
plt.figure(figsize=(8, 6))  # ตั้งขนาดของรูป
plt.contourf(x1_grid, x2_grid, predictions, levels=[-0.5, 0.5, 1.5],  # วาด Decision Boundary
             colors=['red', 'blue'], alpha=0.5)  # สีแดงสำหรับ Class A, สีฟ้าสำหรับ Class B
plt.scatter(X_scaled[:100, 0], X_scaled[:100, 1],  # พล็อตข้อมูล Class A
            color='purple', label='Class A', alpha=1, edgecolor='k')  # จุด Class A
plt.scatter(X_scaled[100:, 0], X_scaled[100:, 1],  # พล็อตข้อมูล Class B
            color='yellow', label='Class B', alpha=1, edgecolor='k')  # จุด Class B
plt.xlabel('Feature x1 (scaled)')  # ชื่อแกน x
plt.ylabel('Feature x2 (scaled)')  # ชื่อแกน y
plt.title('Decision Boundary with Neural Network')  # ชื่อกราฟ
plt.legend(loc='upper right')  # แสดง legend ที่มุมขวาบน
plt.grid(True)  # เพิ่ม grid lines ในกราฟ
plt.xlim(-3, 3)  # กำหนดช่วงของแกน x
plt.ylim(-3, 3)  # กำหนดช่วงของแกน y
plt.show()  # แสดงกราฟ
