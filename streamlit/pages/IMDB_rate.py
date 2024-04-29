import streamlit as st
from rateScrapper import getRate

st.markdown("# Imdb Scrapper")
st.sidebar.markdown("# Imdb Scrapper")
value = st.text_input('Enter a show name')

st.write(f'the rate of {value} is {getRate(value)}')

