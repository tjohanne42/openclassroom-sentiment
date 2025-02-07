from flask import Flask, jsonify, request
import json
import pickle
import nltk
from nltk.corpus import stopwords as nltk_stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
nltk.download("punkt")
nltk.download('stopwords')
nltk.download('punkt_tab')
from unidecode import unidecode
import string
from bs4 import BeautifulSoup


def prepro_text(text,
                stop_words=None,
                remove_html=True,
                remove_accents=True,
                lower=True,
                replace_html=True,
                replace_hashtag=True,
                remove_stopwords=True,
                remove_punct=True,
                stemming=False,
                stemming_ps=None):
    if stop_words is None:
        stop_words = nltk_stopwords.words('english')
        stop_words = list(set(stop_words))
        stop_words.sort()
    # if remove_html:
    #     text = BeautifulSoup(text).text
    if remove_accents:
        text = unidecode(text)
    if lower:
        text = text.lower()
    if replace_hashtag:
        text = text.replace("#", " hashtag ")
    if remove_punct:
        text = text.translate(str.maketrans("", "", string.punctuation))
    if remove_stopwords:
        tokens = word_tokenize(text)
        tokens = [w for w in tokens if not w.lower() in stop_words]
        text = " ".join(tokens)
    text = " ".join(text.split())
    if stemming:
        if stemming_ps is None:
            raise ValueError("Need stemming ps argument.")
        tokens = word_tokenize(text)
        stem_tokens = [stemming_ps.stem(t) for t in tokens]
        text = " ".join(stem_tokens)
    return text


def pred_text(text, encoder, model, stemming_ps, stop_words, stemming=True):
    text = prepro_text(text, stop_words=stop_words, stemming=stemming, stemming_ps=stemming_ps)
    tokens = encoder.transform([text])
    pred = model.predict(tokens)[0]
    pred = int(pred)
    return pred


# Load stuff
model_path = 'sentiment_model.p'
encoder_path = 'sentiment_encoder.pk'
model = pickle.load(open(model_path, 'rb'))
encoder = pickle.load(open(encoder_path, 'rb'))
stemming_ps = PorterStemmer()
stop_words = nltk_stopwords.words('english')
stop_words = list(set(stop_words))
stop_words.sort()

pred = pred_text("This is a test !!! I hate chocolate very much", encoder, model, stemming_ps, stop_words)
print("pred:", pred)

# create Flask app
app = Flask(__name__)


@app.route('/')
def home():
    return "Hello, Heroku!"

    
@app.route('/pred', methods=['POST'])
def pred():
    data = request.json
    if "text" not in data:
        return jsonify({"error": "Need text key"}), 400
    pred = pred_text(data["text"], encoder, model, stemming_ps, stop_words)
    return jsonify({"sentiment": pred}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)
