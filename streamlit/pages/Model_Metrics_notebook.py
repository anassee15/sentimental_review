import streamlit as st
from PIL import Image

st.markdown("# Model metrics run on big data 💽")
st.sidebar.markdown("# Model metrics run on big data 💽")


st.markdown("## Bert")
st.markdown("### Classification report")
image = Image.open('streamlit/src/classification_report/bert_imdb.png')
st.image(image, caption='Bert trained on IMDB')
st.markdown("### Confusion Matrix")
image = Image.open('streamlit/src/confusion_matrix/bert_imdb.png')
st.image(image, caption='Bert trained on IMDB')

st.markdown("### Classification report (twitter)")
image = Image.open('streamlit/src/classification_report/bert_twitter.png')
st.image(image, caption='Bert trained on twitter')
st.markdown("### Confusion Matrix (twitter)")
image = Image.open('streamlit/src/confusion_matrix/bert_twitter.png')
st.image(image, caption='Bert trained on twitter')

st.markdown("## Bert4")
st.markdown("### Classification report")
image = Image.open('streamlit/src/classification_report/bert4_imdb.png')
st.image(image, caption='Bert4 trained on IMDB')
st.markdown("### Confusion Matrix")
image = Image.open('streamlit/src/confusion_matrix/bert4_imdb.png')
st.image(image, caption='Bert4 trained on IMDB')


st.markdown("## Distilbert")
st.markdown("### Classification report")
image = Image.open('streamlit/src/classification_report/distilbert_imdb.png')
st.image(image, caption='Distilbert trained on IMDB')
st.markdown("### Confusion Matrix")
image = Image.open('streamlit/src/confusion_matrix/distilbert_imdb.png')
st.image(image, caption='Distilbert trained on IMDB')

st.markdown("### Classification report (twitter)")
image = Image.open('streamlit/src/classification_report/distilbert_twitter.png')
st.image(image, caption='Distilbert trained on twitter')
st.markdown("### Confusion Matrix (twitter)")
image = Image.open('streamlit/src/confusion_matrix/distilbert_twitter.png')
st.image(image, caption='Distilbert trained on twitter')

st.markdown("## Distilbert4")
st.markdown("### Classification report")
image = Image.open('streamlit/src/classification_report/distilbert4_imdb.png')
st.image(image, caption='Distilbert4 trained on IMDB')
st.markdown("### Confusion Matrix")
image = Image.open('streamlit/src/confusion_matrix/distilbert4_imdb.png')
st.image(image, caption='Distilbert4 trained on IMDB')

st.markdown("## TFIDF")
st.markdown("### Classification report")
image = Image.open('streamlit/src/classification_report/tfidf_imdb.png')
st.image(image, caption='TFIDF trained on IMDB')
st.markdown("### Confusion Matrix")
image = Image.open('streamlit/src/confusion_matrix/tfidf_imdb.png')
st.image(image, caption='TFIDF trained on IMDB')

st.markdown("### Classification report (twitter)")
image = Image.open('streamlit/src/classification_report/tfidf_twitter.png')
st.image(image, caption='TFIDF trained on twitter')
st.markdown("### Confusion Matrix (twitter)")
image = Image.open('streamlit/src/confusion_matrix/tfidf_twitter.png')
st.image(image, caption='TFIDF trained on twitter')

st.markdown("## Count vectorizer")
st.markdown("### Classification report")
image = Image.open('streamlit/src/classification_report/countvectorizer_imdb.png')
st.image(image, caption='Count vectorizer trained on IMDB')
st.markdown("### Confusion Matrix")
image = Image.open('streamlit/src/confusion_matrix/countvectorizer_imdb.png')
st.image(image, caption='Count vectorizer trained on IMDB')

st.markdown("### Classification report (twitter)")
image = Image.open('streamlit/src/classification_report/countvectorizer_twitter.png')
st.image(image, caption='Count vectorizer trained on twitter')
st.markdown("### Confusion Matrix (twitter)")
image = Image.open('streamlit/src/confusion_matrix/countvectorizer_twitter.png')
st.image(image, caption='Count vectorizer trained on twitter')


