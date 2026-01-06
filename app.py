import streamlit as st
import pickle

model = pickle.load(open("spam_knn.pkl","rb"))
tfidf = pickle.load(open("spam_vector.pkl","rb"))

st.title("📧 Spam Classifier (KNN)")

msg = st.text_area("Enter Message")

if st.button("Predict"):
    X = tfidf.transform([msg])
    pred = model.predict(X)[0]
    st.success("Spam" if pred==1 else "Not Spam")
