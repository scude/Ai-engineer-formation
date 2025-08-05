# Interface Streamlit pour tester l’API de prédiction
import streamlit as st
import requests

st.title("Analyse de Sentiment d’un Tweet")

tweet = st.text_input("Saisis ton tweet ici")
if st.button("Analyser"):
    if tweet:
        res = requests.post("http://localhost:8000/predict", params={"tweet": tweet})
        st.write("Sentiment prédit :", res.json().get("sentiment"))
