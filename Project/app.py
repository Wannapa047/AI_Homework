from flask import Flask, request, jsonify
import torch

# โหลดโมเดลที่บันทึกไว้
agent = torch.load('q_learning_agent.pth')

app = Flask(__name__)

@app.route('/translate', methods=['POST'])
def translate():
    # รับข้อความภาษา Karen จากผู้ใช้
    karen_text = request.json.get('karen_text')
    
    # เลือกคำแปลจากโมเดล
    translated_text = agent.select_action(karen_text)
    
    return jsonify({"translated_text": translated_text})

if __name__ == "__main__":
    app.run(debug=True)
