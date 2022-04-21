import streamlit as st
import pandas as pd
import pickle
from PIL import Image

model = pickle.load(open(r"C:\Users\HP\PycharmProjects\streamlit_learn\IRIS-model.pkl", "rb"))

st.header("Iris Classification:")
image = Image.open(r"C:\Users\HP\PycharmProjects\streamlit_learn\wallpaperflare.com_wallpaper.jpg")
st.image(image, use_column_width=True, output_format='JPEG')
st.write("Please insert values, to get Iris class prediction")

SepalLengthCm = st.slider('SepalLengthCm:', 2.0, 6.0)
SepalWidthCm = st.slider('SepalWidthCm:', 0.0, 5.0)
PetalLengthCm = st.slider('PetalLengthCm:', 0.0, 3.0)
PetalWidthCm = st.slider('PetalWidthCm:', 0.0, 2.0)

data = {'SepalLengthCm':SepalLengthCm,
        'SepalWidthCm:':SepalWidthCm,
        'PetalLengthCm:':PetalLengthCm,
        'PetalWidthCm':PetalWidthCm
        }

features = pd.DataFrame(data, index=[0])

pred_proba = model.predict_proba(features)
#or
prediction = model.predict(features)

st.subheader("Prediction Percentages:")
st.write('**Probability of Iris Class being Iris-Setosa is (in %)**:',pred_proba[0][0]*100)
st.write('**Probability of Iris Class being Iris-Versicolor is (in %)**:',pred_proba[0][1]*100)
st.write('**Probability of Iris Class bring Iris-virginica (in %)**:',pred_proba[0][2]*100)