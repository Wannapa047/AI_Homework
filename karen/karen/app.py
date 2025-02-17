from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)

# โหลดโมเดลที่เทรนไว้
model = tf.keras.models.load_model("models/karen_translation_model.keras")

# โหลด Tokenizer
with open("tokenizer/thai_tokenizer.pkl", "rb") as f:
    thai_tokenizer = pickle.load(f)

with open("tokenizer/karen_tokenizer.pkl", "rb") as f:
    karen_tokenizer = pickle.load(f)

max_length = 20

# ฟังก์ชันแปลภาษา
def translate(sentence, tokenizer, max_length):
    sequence = tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(sequence, maxlen=max_length, padding="post")
    prediction = model.predict([padded, padded])
    predicted_index = np.argmax(prediction[0], axis=-1)
    translated_words = [tokenizer.index_word[idx] for idx in predicted_index if idx > 0]
    return " ".join(translated_words)

@app.route('/')
def index():
    return render_template('index.html')

# สร้าง API Endpoint
@app.route("/translate", methods=["POST"])
def translate_text():
    data = request.get_json()
    input_text = data.get("text", "")
    lang = data.get("direction", "karen-to-thai")

    if lang == "thai-to-karen":
        translated_text = translate(input_text, karen_tokenizer, max_length)
    elif (lang == "karen-to-thai"):
        translated_text = translate(input_text, thai_tokenizer, max_length)

    return jsonify({"translated_text": translated_text})

if __name__ == "__main__":
    app.run(debug=True)