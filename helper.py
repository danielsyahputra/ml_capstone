import json
import numpy as np
import tensorflow as tf
import pandas as pd

from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

def get_padded_text(tokenizer, desc_cleaned):
    max_length = 100
    trunc_type = 'post'
    padding_type = 'post'

    pred_sequences = tokenizer.texts_to_sequences([desc_cleaned])
    pred_padded = pad_sequences(pred_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    return pred_padded

def load_stuff():

    # Load tokenizer
    with open('assets/tokenizer.json') as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
    
    # Load model
    model = tf.keras.models.load_model('assets/model.h5')

    # Load encoder
    encoder = LabelEncoder()
    encoder.classes_ = np.load('assets/encoder.npy', allow_pickle=True)

    # Load description
    description = pd.read_pickle("data/description.pkl")

    return tokenizer, model, encoder, description

def get_prediction(model, pred_padded, encoder, description):
    result = model.predict(pred_padded)[0]
    idx = result.argsort()[-3:][::-1]

    conditions = encoder.inverse_transform(idx)

    result_json = []
    for i in range(len(conditions)):
        condition = conditions[i]
        probability = float(result[idx[i]])
        deskripsi = description[description['Condition'] == condition]['Deskripsi'].values[0]
        result_json.append(
            {
                "disease": "Fibromyalgia" if condition == 'ibromyalgia' else condition.title(),
                "probability": probability,
                "deskripsi": deskripsi,
            }
        )
    return result_json

API_DESC = description = """
> Disease Prediction API helps you do awesome stuff. 🚀

## Our Team
- Daniel Syahputra Purba
- Mardianto
- Larasati

You will be able to:

* **Predict disease based on symptoms** (Ongoing).
"""