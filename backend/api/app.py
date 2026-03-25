from flask import Flask, request, jsonify
from flask_cors import CORS
from model import predict_sentiment
import pandas as pd   

app = Flask(__name__)
CORS(app)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    result = predict_sentiment(text)

    return jsonify({
        "prediction": result
    })

#dataset
@app.route('/stats', methods=['GET'])
def stats():
    data = pd.read_csv("C:/Users/luban/smart-feedback-analyzer/dataset/IMDB Dataset.csv")

    total = len(data)
    positive = len(data[data['sentiment'] == 'positive'])
    negative = len(data[data['sentiment'] == 'negative'])

    return jsonify({
        "total": total,
        "positive": positive,
        "negative": negative
    })

if __name__ == '__main__':
    app.run(debug=True)