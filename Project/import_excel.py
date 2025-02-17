import pandas as pd

# โหลดไฟล์ CSV
file_path = "karen_words.csv"
df = pd.read_csv(file_path, sep="\t", header=None)

# ตั้งชื่อคอลัมน์ให้ถูกต้อง
df.columns = ["karen_text", "thai_text", "pronunciation", "confidence"]

# แปลงค่า confidence เป็นตัวเลข
df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

# แสดงตัวอย่างข้อมูล
print(df.head())
