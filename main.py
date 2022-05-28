from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

from preprocess import SpacyPreprocessor

spacy_model = SpacyPreprocessor.load_model()
preprocessor = SpacyPreprocessor(spacy_model=spacy_model, lemmatize=True, remove_numbers=True, remove_stopwords=True)

class Teks(BaseModel):
    desc: str

@app.get("/")
async def root():
    print("Preprocessor Imported!")
    return {"message": "Hello World"}

@app.post("/predict/")
async def predict(teks: Teks):
    desc_cleaned = preprocessor.preprocess_text(teks.desc)
    return {
            "desc": desc_cleaned,
            "disease": "Depression",
        }