from fastapi import FastAPI
from pydantic import BaseModel
import translators as ts

from preprocess import SpacyPreprocessor
from helper import *


spacy_model = SpacyPreprocessor.load_model()
preprocessor = SpacyPreprocessor(spacy_model=spacy_model, lemmatize=True, remove_numbers=True, remove_stopwords=True)

tokenizer, model, encoder, description = load_stuff()

app = FastAPI()

class Teks(BaseModel):
    desc: str

@app.get("/")
async def root():
    return {"message": "Please, go to /docs endpoint for more info"}

@app.post("/predict/")
async def predict(teks: Teks):

    # Translate teks into english first
    translated_text = ts.google(teks.desc, to_language='en')

    # Preprocess
    desc_cleaned = preprocessor.preprocess_text(translated_text)

    # Get padded text
    pred_padded = get_padded_text(tokenizer, desc_cleaned)

    # Predict
    result = get_prediction(model, pred_padded, encoder, description)

    return {
        "description": desc_cleaned,
        "result": result
    }