import numpy as np
from PIL import Image,  ImageOps
import streamlit as st
import tensorflow as tf

def load_model():
    model = tf.keras.models.load_model('Model_Kopi_Epoch_75.h5')
    return model

model = load_model()

st.title("Klasifikasi Biji Kopi")

file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])


def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction


if file is None:
    st.text("Silahkan upload gambar")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    labels = ['Dark', 'Green','Light', 'Medium']
    string = "Prediksi Gambar : "+labels[np.argmax(predictions)]
    if predictions[0][0] == 0:
        print()
    elif predictions[0][1] != 0:
        print()
    elif predictions[0][2] != 0:
        print()
    else:
        print()
    st.success(string)