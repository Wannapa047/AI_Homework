import pandas as pd  # สำหรับจัดการข้อมูล
import numpy as np   # สำหรับการคำนวณทางคณิตศาสตร์
import matplotlib.pyplot as plt  # สำหรับการแสดงกราฟ 

# ฟังก์ชันสำหรับคำนวณสมการ Regression
def calculate_regression(df):
    """
    คำนวณสมการเส้นถดถอยจาก DataFrame
    df: DataFrame ที่มีคอลัมน์ 'x' และ 'y'
    return: slope (ความชัน), intercept (จุดตัดแกน y)
    """
    x_mean = df['x'].mean()
    y_mean = df['y'].mean()
    numerator = ((df['x'] - x_mean) * (df['y'] - y_mean)).sum()
    denominator = ((df['x'] - x_mean) ** 2).sum()
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    return slope, intercept

# ฟังก์ชันสำหรับการทำนายค่า y จาก x ใหม่
def predict(x_new, slope, intercept):
    """
    ทำนายค่า y จากสมการเส้นถดถอย
    x_new: ค่า x ใหม่ที่ต้องการทำนาย
    slope: ความชันของสมการ
    intercept: จุดตัดแกน y ของสมการ
    return: ค่าที่คาดการณ์ของ y
    """
    return slope * x_new + intercept

# กำหนดข้อมูล
data = {
    'x': [29, 28, 34, 31, 25],  # อุณหภูมิสูงสุด (°C)
    'y': [77, 62, 93, 84, 59]   # จำนวนสั่งซื้อชาเย็น
}

# สร้าง DataFrame จากข้อมูลที่กำหนด
df = pd.DataFrame(data)

# ตรวจสอบข้อมูลเบื้องต้น
print("กำหนดข้อมูล :")
print(df)

# คำนวณสมการ Regression
slope, intercept = calculate_regression(df)
print(f"\nสมการเส้นถดถอย : y = {slope:.2f}x{intercept:.2f}")

# การทำนายค่า
x_new = 30  # ตัวอย่างค่า x ใหม่
y_pred = predict(x_new, slope, intercept)
print(f"\nค่า y ที่คาดการณ์คือ : {y_pred:.2f}")

# การแสดงกราฟ 
plt.scatter(df['x'], df['y'], color='blue', label='Data points')  # จุดข้อมูล
plt.plot(df['x'], slope * df['x'] + intercept, color='red', label='Regression line')  # เส้นถดถอย
plt.xlabel('x (High temperature in °C)')
plt.ylabel('y (Iced tea orders)')
plt.title('Linear Regression')
plt.legend()
plt.show()
