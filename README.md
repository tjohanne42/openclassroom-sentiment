# openclassroom-sentiment

Pipeline to train different models for sentiment classification (main.ipynb).  
Using Flask API to deploy the selected model (app.py).

# Table of content:
- main.ipynb: pipeline to train models with mlflow
- app.py: flask API to use trained model
- sentiment_encoder.pk, sentiment_model.p: model files for API
- test_api.py: unitary test for API
- nltk.txt, Procfile: setup files to deploy on Heroku

# Train models:
``` bash
pip install jupyter jupyterlab
jupyter lab
# use main.ipynb
```

# Use API:
``` bash
pip install -r requirements.txt
python app.py
```

# Test API:
``` bash
pip install -r requirements.txt
pytest test_api.py
```