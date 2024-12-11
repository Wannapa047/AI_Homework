import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# สร้างข้อมูลสำหรับ Class 1 และ Class 2
x1, y1 = make_blobs(n_samples=100,
                   n_features=2,
                   centers=1,
                   center_box=(2.0, 2.0),
                   cluster_std=0.23,
                   random_state=64)

x2, y2 = make_blobs(n_samples=100,
                   n_features=2,
                   centers=1,
                   center_box=(3.0, 3.0),
                   cluster_std=0.23,
                   random_state=64)

# รวมข้อมูลทั้งหมดเพื่อทำการปรับขนาด
X = np.vstack((x1, x2))

# ใช้ StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# แยกข้อมูลที่ปรับขนาดแล้วกลับออกเป็น x1 และ x2
x1_scaled = X_scaled[:100, :]
x2_scaled = X_scaled[100:, :]

# นิยามฟังก์ชันตัดสินใจ (decision function)
def decision_function(x1, x2):
    return x1 + x2 - 0

# สร้าง grid สำหรับ decision boundary
x1_range = np.linspace(-3, 3, 500)
x2_range = np.linspace(-3, 3, 500)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

# คำนวณค่าฟังก์ชันตัดสินใจ
g_values = decision_function(x1_grid, x2_grid)

# พล็อต Decision Plane รวมกับจุดข้อมูล
plt.figure(figsize=(8, 6))
plt.contourf(x1_grid, x2_grid, g_values, levels=[-np.inf, 0, np.inf], colors=['red', 'blue'], alpha=0.5)
plt.contour(x1_grid, x2_grid, g_values, levels=[0], colors='black', linewidths=2)  # เส้นแบ่ง
plt.scatter(x1_scaled[:, 0], x1_scaled[:, 1], color='purple', s=60, label='Class 1', alpha=1)
plt.scatter(x2_scaled[:, 0], x2_scaled[:, 1], color='yellow', s=60, label='Class 2', alpha=1)
plt.xlabel('Feature x1')
plt.ylabel('Feature x2')
plt.title('Decision Plane')
plt.legend(loc='upper right')
plt.grid(True)
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.show()