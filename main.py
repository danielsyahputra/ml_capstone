from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import translators as ts

from preprocess import SpacyPreprocessor
from helper import *


spacy_model = SpacyPreprocessor.load_model()
preprocessor = SpacyPreprocessor(spacy_model=spacy_model, lemmatize=True, remove_numbers=True, remove_stopwords=True)

tokenizer, model, encoder, description = load_stuff()

app = FastAPI(
    title="Disease Prediction API",
    description=API_DESC,
    version="0.0.1",
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    }
)

class Teks(BaseModel):
    desc: str

@app.get("/", tags=["General"], response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>Disease Prediction API</title>
        </head>
        <body>
            <center>
                <h1>Disease Prediction API</h1>
                <h3>Please, go to <a href='https://ml.matthewbd.my.id/docs'>API Documentation</a> for more information.</h3>
            </center>
        </body>
    </html>
    """

@app.get("/disease", tags=["General"])
async def root():
    return json.loads(description.to_json(orient="records"))


@app.post("/predict/", tags=['Prediction'])
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