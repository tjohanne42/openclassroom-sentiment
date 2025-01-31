# pip install opencensus-ext-azure
# https://portal.azure.com/?quickstart=true#@theojohannetgmail.onmicrosoft.com/resource/subscriptions/fb570171-2661-474b-98f3-f151bd97898e/resourcegroups/conteneur1/providers/microsoft.insights/components/insight1/overview
from flask import Flask, jsonify, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import logging
from opencensus.ext.azure.log_exporter import AzureLogHandler


# define model_pred func
def model_pred(text):
    global pipe
    pred = pipe(text)[0]
    pred = int("".join([c for c in pred["label"] if c.isnumeric()]))
    return pred
model_name = 'phanerozoic/BERT-Sentiment-Classifier'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)

# create Flask app
app = Flask(__name__)

logger = logging.getLogger(__name__)
logger.addHandler(AzureLogHandler(connection_string="InstrumentationKey=eb1f5681-33d5-4303-a67b-ced937b5ad09;IngestionEndpoint=https://francecentral-1.in.applicationinsights.azure.com/;LiveEndpoint=https://francecentral.livediagnostics.monitor.azure.com/;ApplicationId=9d966a96-db65-43fc-a9ef-b24d87e21088"))
logger.setLevel(logging.INFO)
logger.info('Hello, World!')


@app.route('/pred', methods=['POST'])
def pred():
    data = request.json
    if "text" not in data:
        return jsonify({"error": "Need text key"}), 400
    pred = model_pred(data["text"])
    return jsonify({"sentiment": pred}), 200


@app.route('/logprederror', methods=['POST'])
def logprederror():
    data = request.json
    for k in ["text", "sentiment"]:
        if k not in data:
            return jsonify({"error": f"Need '{k}' key"}), 400
    text = f"{data['text']}❣︎{data['sentiment']}"
    print("log text:", text)
    logger.warning(text)
    return jsonify({"status": "success"}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)