from flask import Flask, request, jsonify
from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Initialize Flask app
app = Flask(__name__)

# Load model and tokenizer once at startup
#model_path = r"C:\Users\harri\OneDrive\Desktop\Thesis\my_bert_model_latest_22_06"
model_path = "my_bert_model_latest_22_06"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
model.eval()

# Prediction function
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits.softmax(dim=-1)
        prediction = probs.argmax().item()
        confidence = probs.max().item()
    return prediction, confidence

# Routes
@app.route("/")
def index():
    return "BERT Model API is running!"

@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json(force=True)

    if "text" not in data:
        return jsonify({"error": "Missing 'text' in request"}), 400

    text = data["text"]
    prediction, confidence = predict(text)

    return jsonify({
        "ismeetinginvite": True if prediction == 1 else False,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=True)
