import gradio as gr
import pandas as pd
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("churn_model.h5")

def predict_churn(*inputs):
    # Build DataFrame as in the previous example, or turn inputs into a numpy array if that's what your model needs
    input_df = ... # same as above
    proba = model.predict(input_df)[0][0]
    return "Churn" if proba > 0.5 else "Not Churn"

# (Rest of code as above)
