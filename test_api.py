import pytest
import json
from flask import Flask
from app import app
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords as nltk_stopwords

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

# Test de la page d'accueil
def test_home(client):
    response = client.get("/")
    assert response.status_code == 200

# Test de la route /pred avec une entrée valide
def test_prediction_valid(client):
    response = client.post("/pred", json={"text": "I love this product!"})
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "sentiment" in data
    assert isinstance(data["sentiment"], int)
    assert data["sentiment"] >= 0
    assert data["sentiment"] <= 1

def test_prediction_no_text(client):
    response = client.post("/pred", json={})
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "error" in data
