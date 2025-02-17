"""
import pandas as pd

# โหลดไฟล์ CSV
df = pd.read_csv(r"C:\Users\Msi01\OneDrive - Chitralada Technology Institute\AI\Project\karen_words.csv")

# แสดงตัวอย่างข้อมูล
print(df.head())
"""

import re
import pandas as pd

# ฟังก์ชันในการแยกคำด้วยวิธีการพื้นฐาน
def simple_tokenizer(text):
    return re.findall(r'\w+', text)  # ใช้ regular expression เพื่อแยกคำ

# อ่านไฟล์ CSV
df = pd.read_csv("C:\Users\Msi01\OneDrive - Chitralada Technology Institute\AI\Project\Q-Learning.py")

# ตัวอย่างการแปลงข้อความจาก DataFrame เป็นคำ
sample_text = df["karen_text"][0]
tokens = simple_tokenizer(sample_text)

# แสดงผล
print(tokens)

