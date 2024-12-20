from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# Load the sentiment analysis pipeline
model_pipeline = pipeline("text-classification", model="Menna-Khaled/SentimentAnalysis")

@app.route('/')
def index():
    return render_template('index.html', input_text="")

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form.get('inputText')
    if text:
        # Perform sentiment analysis
        result = model_pipeline(text)
        sentiment = result[0]['label']
        score = result[0]['score']
        return render_template('index.html', input_text=text, sentiment=sentiment, score=score)
    else:
        return render_template('index.html', input_text="", error="Please enter text to analyze.")

@app.route('/clear', methods=['POST'])
def clear():
    return render_template('index.html', input_text="")

if __name__ == "__main__":
    app.run(debug=True)