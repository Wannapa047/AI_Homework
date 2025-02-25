<<<<<<< HEAD
import pickle
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np

# โหลดไฟล์ Excel
file_path = "dataset/Karen_words.xlsx"
df = pd.read_excel(file_path)

# ทำความสะอาดชื่อคอลัมน์
df.columns = df.columns.str.strip()

# ตรวจสอบค่าที่หายไป
df = df.dropna()

# แปลงข้อมูลเป็นข้อความ (แก้ไขปัญหา int object)
df['thai_text'] = df['thai_text'].astype(str)
df['karen_text'] = df['karen_text'].astype(str)

# แบ่งข้อมูลเป็น train (80%) และ temp (20%)
train_data, temp_data = train_test_split(df, test_size=0.2, random_state=42)

# แบ่ง temp เป็น validation (10%) และ test (10%)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Tokenization
num_words = 5000
max_length = 20

thai_tokenizer = Tokenizer(num_words=num_words)
karen_tokenizer = Tokenizer(num_words=num_words)

thai_tokenizer.fit_on_texts(train_data['thai_text'])
karen_tokenizer.fit_on_texts(train_data['karen_text'])

thai_sequences = thai_tokenizer.texts_to_sequences(train_data['thai_text'])
karen_sequences = karen_tokenizer.texts_to_sequences(train_data['karen_text'])

thai_padded = pad_sequences(thai_sequences, maxlen=max_length, padding='post')
karen_padded = pad_sequences(karen_sequences, maxlen=max_length, padding='post')

# สร้างโมเดล Encoder-Decoder
latent_dim = 256

encoder_inputs = Input(shape=(max_length,))
enc_emb = Embedding(num_words, latent_dim, mask_zero=True)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(max_length,))
dec_emb = Embedding(num_words, latent_dim, mask_zero=True)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(num_words, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# เทรนโมเดล
epochs = 50
batch_size = 64
model.fit([thai_padded, karen_padded], karen_padded, batch_size=batch_size, epochs=epochs, validation_split=0.1)

model.save('../models/karen_translation_model.keras')
pickle.dump(thai_tokenizer, open("../tokenizer/thai_tokenizer.pkl", "wb"))
pickle.dump(karen_tokenizer, open("../tokenizer/karen_tokenizer.pkl", "wb"))

# ทดสอบการแปล
def translate(sentence, tokenizer, max_length):
    sequence = tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(sequence, maxlen=max_length, padding='post')
    prediction = model.predict([padded, padded])
    predicted_index = np.argmax(prediction[0], axis=-1)
    translated_words = [tokenizer.index_word[idx] for idx in predicted_index if idx > 0]
    return ' '.join(translated_words)

# ตัวอย่างการแปล
example_sentence = "အီၣ်"
translated_sentence = translate(example_sentence, thai_tokenizer, max_length)
print(f"แปล: {translated_sentence}")

# บันทึกข้อมูลเป็น CSV
train_data.to_csv("dataset/train_data.csv", index=False, encoding='utf-8')
val_data.to_csv("dataset/val_data.csv", index=False, encoding='utf-8')
test_data.to_csv("dataset/test_data.csv", index=False, encoding='utf-8')

# บันทึกข้อมูลเป็น JSON
train_data.to_json("dataset/train_data.json", orient="records", force_ascii=False)
val_data.to_json("dataset/val_data.json", orient="records", force_ascii=False)
test_data.to_json("dataset/test_data.json", orient="records", force_ascii=False)

=======
import pickle
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np

# โหลดไฟล์ Excel
file_path = "dataset/Karen_words.xlsx"
df = pd.read_excel(file_path)

# ทำความสะอาดชื่อคอลัมน์
df.columns = df.columns.str.strip()

# ตรวจสอบค่าที่หายไป
df = df.dropna()

# แปลงข้อมูลเป็นข้อความ (แก้ไขปัญหา int object)
df['thai_text'] = df['thai_text'].astype(str)
df['karen_text'] = df['karen_text'].astype(str)

# แบ่งข้อมูลเป็น train (80%) และ temp (20%)
train_data, temp_data = train_test_split(df, test_size=0.2, random_state=42)

# แบ่ง temp เป็น validation (10%) และ test (10%)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Tokenization
num_words = 5000
max_length = 20

thai_tokenizer = Tokenizer(num_words=num_words)
karen_tokenizer = Tokenizer(num_words=num_words)

thai_tokenizer.fit_on_texts(train_data['thai_text'])
karen_tokenizer.fit_on_texts(train_data['karen_text'])

thai_sequences = thai_tokenizer.texts_to_sequences(train_data['thai_text'])
karen_sequences = karen_tokenizer.texts_to_sequences(train_data['karen_text'])

thai_padded = pad_sequences(thai_sequences, maxlen=max_length, padding='post')
karen_padded = pad_sequences(karen_sequences, maxlen=max_length, padding='post')

# สร้างโมเดล Encoder-Decoder
latent_dim = 256

encoder_inputs = Input(shape=(max_length,))
enc_emb = Embedding(num_words, latent_dim, mask_zero=True)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(max_length,))
dec_emb = Embedding(num_words, latent_dim, mask_zero=True)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(num_words, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# เทรนโมเดล
epochs = 50
batch_size = 64
model.fit([thai_padded, karen_padded], karen_padded, batch_size=batch_size, epochs=epochs, validation_split=0.1)

model.save('../models/karen_translation_model.keras')
pickle.dump(thai_tokenizer, open("../tokenizer/thai_tokenizer.pkl", "wb"))
pickle.dump(karen_tokenizer, open("../tokenizer/karen_tokenizer.pkl", "wb"))

# ทดสอบการแปล
def translate(sentence, tokenizer, max_length):
    sequence = tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(sequence, maxlen=max_length, padding='post')
    prediction = model.predict([padded, padded])
    predicted_index = np.argmax(prediction[0], axis=-1)
    translated_words = [tokenizer.index_word[idx] for idx in predicted_index if idx > 0]
    return ' '.join(translated_words)

# ตัวอย่างการแปล
example_sentence = "အီၣ်"
translated_sentence = translate(example_sentence, thai_tokenizer, max_length)
print(f"แปล: {translated_sentence}")

# บันทึกข้อมูลเป็น CSV
train_data.to_csv("dataset/train_data.csv", index=False, encoding='utf-8')
val_data.to_csv("dataset/val_data.csv", index=False, encoding='utf-8')
test_data.to_csv("dataset/test_data.csv", index=False, encoding='utf-8')

# บันทึกข้อมูลเป็น JSON
train_data.to_json("dataset/train_data.json", orient="records", force_ascii=False)
val_data.to_json("dataset/val_data.json", orient="records", force_ascii=False)
test_data.to_json("dataset/test_data.json", orient="records", force_ascii=False)

>>>>>>> e579062612ffcaf57ce4647621a9d25e31122bd3
print("Model training completed and translation test done.")