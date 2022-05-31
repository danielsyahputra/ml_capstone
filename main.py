from fastapi import FastAPI
from pydantic import BaseModel

import tensorflow as tf
import json
import numpy as np
import os
import pandas as pd

from preprocess import SpacyPreprocessor
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder


spacy_model = SpacyPreprocessor.load_model()
preprocessor = SpacyPreprocessor(spacy_model=spacy_model, lemmatize=True, remove_numbers=True, remove_stopwords=True)

app = FastAPI()


class Teks(BaseModel):
    desc: str

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict/")
async def predict(teks: Teks):
    desc_cleaned = preprocessor.preprocess_text(teks.desc)
    print(os.getcwd())

    with open('assets/tokenizer.json') as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
   
    max_length = 100
    trunc_type = 'post'
    padding_type = 'post'

    pred_sequences = tokenizer.texts_to_sequences([desc_cleaned])
    pred_padded = pad_sequences(pred_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    model = tf.keras.models.load_model('assets/model.h5')

    result = model.predict(pred_padded)[0]
    idx = result.argsort()[-3:][::-1]

    encoder = LabelEncoder()
    encoder.classes_ = np.load('assets/encoder.npy', allow_pickle=True)

    conditions = encoder.inverse_transform(idx)

    ############################################

    # load description and medicine
    description = pd.read_pickle("data/description.pkl")
    medicine = pd.read_pickle("data/medicine.pkl")

    result_json = []
    for condition in conditions:
        deskripsi = description[description['Condition'] == condition]['Deskripsi'].values[0]
        obats = medicine.loc[condition].sort_values(['usefulCount'], ascending=False).nlargest(3, ['usefulCount']).index.tolist()
        result_json.append(
            {
                "disease": condition,
                "deskripsi": deskripsi,
                "obat": obats
            }
        )

    return {
        "result": result_json
    }