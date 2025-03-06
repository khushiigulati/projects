from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask_cors import CORS
import torch

app = Flask(__name__)
CORS(app)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Predefined responses
predefined_responses = {
    "hello": "Hi there! How can I assist you today?",
    "help": "I'm here to help! What do you need assistance with?",
    "bye": "Goodbye! Have a great day!",
    "your name":"I am aria,a virtual chat bot!how can I help you?",
    "who are you":"I am aria,a virtual chat bot!how can I help you?"

}

# Function to generate a response
def generate_response(input_text, history=None):
    if history is None:
        history = []

    new_user_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = new_user_input_ids

    if history:
        bot_input_ids = torch.cat([history, new_user_input_ids], dim=-1)

    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, chat_history_ids

# Function to check for predefined responses
def check_predefined_responses(user_input):
    for keyword, response in predefined_responses.items():
        if keyword in user_input.lower():
            return response
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    if user_input:
        predefined_response = check_predefined_responses(user_input)
        if predefined_response:
            response = predefined_response
        else:
            response, _ = generate_response(user_input)
        return jsonify({"response": response})
    return jsonify({"response": "I didn't understand that. Can you please rephrase?"})

if __name__ == '__main__':
    app.run(debug=True)
