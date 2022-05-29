from fastapi import FastAPI
from pydantic import BaseModel

import tensorflow as tf
import json
import numpy as np
import os

from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

from preprocess import pre_process

app = FastAPI()


class Teks(BaseModel):
    desc: str

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict/")
async def predict(teks: Teks):
    desc_cleaned = teks.desc
    print(os.getcwd())

    with open('assets/tokenizer_with_counts_100.json') as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
   
    max_length = 100
    trunc_type = 'post'
    padding_type = 'post'

    pred_sequences = tokenizer.texts_to_sequences([desc_cleaned])
    pred_padded = pad_sequences(pred_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    model = tf.keras.models.load_model('assets/GRU_dense_with_count_100.h5')

    result = model.predict(pred_padded)[0]
    idx = np.argmax(result)

    encoder = LabelEncoder()
    encoder.classes_ = np.load('assets/encoder_with_count_100.npy', allow_pickle=True)

    condition = encoder.inverse_transform([idx])[0]

    return {
            "desc": desc_cleaned,
            "disease": condition,
        }