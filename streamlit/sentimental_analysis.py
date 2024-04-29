import os.path

import streamlit as st

from rateScrapper import getRate
from DataBuilder import DataBuilder

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TextClassificationPipeline
from tfidf import *  

import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS


def rateSystemOutput(results,multiple_label=False):
    somme = 0
    if multiple_label:
        for result in results:
            if result[0]['label'] == 'LABEL_0':
                somme+=1.5
            elif result[0]['label'] == 'LABEL_1':
                somme+=3.5
            elif result[0]['label'] == 'LABEL_2':
                somme+=7.5
            elif result[0]['label'] == 'LABEL_3':
                somme+=9.5
        return somme/len(results)
    else:
        for result in results:
            if result[0]['label'] == 'LABEL_1':
                somme += 1
        return somme/len(results) * 10

def rateSystemOutputList(results):
    somme = 0
    for result in results:
        if result== 1:
            somme += 1
    return somme/len(results) * 10

st.markdown("# Movie/show rate 🎥")
st.sidebar.markdown("# Movie/show rate 🎥")

value = st.text_input('Enter a show name', value='Titanic')
option = st.selectbox('Which model would u like to use ?',('Distilbert uncased', 'Bert cased', 'tfidf ', 'countVectorizer ','Distilbert4 uncased', 'Bert4 cased'))
option = option.lower()
name, para = option.split(' ')

db = DataBuilder(value)


st.markdown('## The rate given by IMDB :')
st.write(f'the rate of {value} is {getRate(value)}')

st.markdown('## The rate given by our model using reaction from Twitter :')


if os.path.exists('model/'+name+'/'+name+'.pkl'):
    if name == 'tfidf':
        pipe = TfidfPipepline(pkl_name='model/'+name+'/'+name+'.pkl')
    else:
        pipe = CountVectorizerPipepline(pkl_name='model/'+name+'/'+name+'.pkl')
    results = db.get_tweets_to_analyze()
    text = " ".join(review for review in results)
    results = pipe._train_model.predict(results)
    st.write(f'the rate of {value} is {round(rateSystemOutputList(results), 2)}')
else:
    multiple_label=False
    model = AutoModelForSequenceClassification.from_pretrained("model/"+name+"/", num_labels=2,ignore_mismatched_sizes=True)
    if name[-1].isdigit():
        name = name.rstrip(name[-1])
        multiple_label=True
    tokenizer = AutoTokenizer.from_pretrained(name+"-base-"+para)
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, top_k=1)
    results = db.get_tweets_to_analyze()
    text = " ".join(review for review in results)
    results = pipe(results)
    st.write(f'the rate of {value} is {round(rateSystemOutput(results,multiple_label), 2)}')



word_cloud = WordCloud(collocations = False, background_color = 'black', stopwords= set(STOPWORDS), width=800, height=400).generate(text)

arr = np.random.normal(1, 1, size=100)

fig, ax = plt.subplots()
ax.imshow(word_cloud, interpolation='bilinear')
ax.title.set_text(f'Word cloud of tweets contains #{value.replace(" ", "")}')
ax.axis("off")

st.pyplot(fig)
