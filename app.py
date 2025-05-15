from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)
model = joblib.load('sentiment_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['user_input'].strip()
    
    # For very short inputs, return neutral directly
    if len(text.split()) < 2:
        prediction = 'neutral'
        confidence = 0.5
    else:
        probs = model.predict_proba([text])[0]
        classes = model.classes_
        max_index = probs.argmax()
        confidence = probs[max_index]
        prediction = classes[max_index]
        # If confidence too low, fallback to neutral
        if confidence < 0.6:
            prediction = 'neutral'

    return render_template('index.html', input=text, result=prediction, confidence=round(confidence*100, 2))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
