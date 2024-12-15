import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# สร้างข้อมูลสำหรับ Data set A และ B
x1, y1 = make_blobs(n_samples=100, n_features=2, centers=[[2.0, 2.0]], cluster_std=0.75, random_state=64)
x2, y2 = make_blobs(n_samples=100, n_features=2, centers=[[3.0, 3.0]], cluster_std=0.75, random_state=64)

# รวมข้อมูลและปรับ Label
X = np.vstack((x1, x2))
y = np.hstack((np.zeros(100), np.ones(100)))  # Class A = 0, Class B = 1

# ใช้ StandardScaler เพื่อปรับขนาดข้อมูล
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled[:, 0] = X_scaled[:, 0] * -1


# แบ่งข้อมูลสำหรับ Train และ Test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.5, random_state=45)

# สร้าง Neural Network
clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=45)
clf.fit(X_train, y_train)

# ประเมินผลการทำงาน
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# สร้าง Grid สำหรับ Decision Boundary
x1_range = np.linspace(-3, 3, 500)
x2_range = np.linspace(-3, 3, 500)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

# คำนวณ Decision Boundary
Z = clf.predict_proba(np.c_[x1_grid.ravel(), x2_grid.ravel()])[:, 1]
Z = Z.reshape(x1_grid.shape)

# พล็อตผลลัพธ์
plt.figure(figsize=(8, 6))
plt.contourf(x1_grid, x2_grid, Z, levels=[0, 0.5, 1], colors=['red', 'blue'], alpha=0.5)
plt.contour(x1_grid, x2_grid, Z, levels=[0.5], colors='black', linewidths=2)  # เส้นแบ่ง
plt.scatter(X_scaled[y == 0][:, 0], X_scaled[y == 0][:, 1], color='red', s=60, label='Class 1', alpha=1)
plt.scatter(X_scaled[y == 1][:, 0], X_scaled[y == 1][:, 1], color='blue', s=60, label='Class 2', alpha=1)
plt.xlabel('Feature x1')
plt.ylabel('Feature x2')
plt.title('Decision Plane')
plt.legend(loc='lower right')
plt.grid(True)
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.show()
