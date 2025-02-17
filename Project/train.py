import pandas as pd
import numpy as np

# โหลดข้อมูลจากไฟล์ CSV
df = pd.read_csv(r"C:\Users\Msi01\OneDrive - Chitralada Technology Institute\AI\Project\karen_words.csv")

# ลบช่องว่างในชื่อคอลัมน์
df.columns = df.columns.str.strip()

# ตรวจสอบชื่อคอลัมน์
print("ชื่อคอลัมน์:", df.columns)

# กำหนด action จากคอลัมน์ 'thai_text'
actions = df["thai_text"].unique()  # คำแปลที่เป็นไปได้ทั้งหมด
print("คำแปลที่เป็นไปได้ทั้งหมด:", actions)

# สร้างฟังก์ชันในการแปลงข้อความเป็นข้อมูลที่โมเดลเข้าใจ
def preprocess_data(text):
    return text.lower()  # แปลงเป็นตัวพิมพ์เล็กเพื่อให้การเรียนรู้ไม่ถูกกระทบจากตัวพิมพ์

# เตรียมข้อมูลสำหรับการฝึกโมเดล
X = df["karen_text"].apply(preprocess_data).values  # ข้อความภาษา Karen
y = df["thai_text"].values  # คำแปลภาษาไทย

# แสดงตัวอย่างข้อมูล
print("\nตัวอย่างข้อมูล:")
for i in range(5):
    print(f"Karen Text: {X[i]}, Thai Text: {y[i]}")

# เพิ่มฟังก์ชันการฝึกด้วย Q-Learning หรืออื่นๆ ที่ต้องการ
# (ตัวอย่างการฝึกจะขึ้นอยู่กับการใช้งาน Reinforcement Learning หรือ Q-Learning ของคุณ)

