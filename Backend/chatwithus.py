from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import openai

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app, resources={r"/chat": {"origins": "*"}})  # Allow all origins

# 🔑 Correct OpenAI API Client Initialization
client = openai.OpenAI(api_key="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")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '').strip()

    if not user_message:
        return jsonify({"reply": "No message received."}), 400

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": user_message}],
            max_tokens=100,
            temperature=0.7
        )
        reply = response.choices[0].message.content.strip()  # ✅ Fix response access
    except Exception as e:
        reply = f"Error generating response: {e}"

    return jsonify({"reply": reply})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # Explicitly set port
