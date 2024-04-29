import streamlit as st

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TextClassificationPipeline

from tfidf import *  

st.markdown("# sentimal analysis on customised sentence 🧑🏾‍⚖️ ")
st.sidebar.markdown("# sentimal analysis on customised sentence 🧑🏾‍⚖️")

option = st.selectbox('Which model would u like to use ?',('Distilbert uncased', 'Bert cased', 'tfidf ', 'countVectorizer ','Distilbert4 uncased', 'Bert4 cased'))
option = option.lower()
name, para = option.split(' ')

value = st.text_input('Enter a sentence to analyze')

is4 = False

def toText(value):
    if type(value) == type(list('')):
        if is4:
            if value[0][0]['label'] == 'LABEL_0':
                result = 'Very negatif'
            elif value[0][0]['label'] == 'LABEL_1':
                result = 'Negatif'
            elif value[0][0]['label'] == 'LABEL_2':
                result = 'Positif'
            else:
                result = 'Very Positif'
        else:
            if value[0][0]['label'] == 'LABEL_0':
                result = 'Negatif'
            else :
                result = 'Positif'
    else:
        if value == 0:
            result = 'Negatif'
        else :
            result = 'Positif'

    return result
    

if os.path.exists('model/'+name+'/'+name+'.pkl'):
    if name == 'tfidf':
        pipe = TfidfPipepline(pkl_name='model/'+name+'/'+name+'.pkl')
    else:
        pipe = CountVectorizerPipepline(pkl_name='model/'+name+'/'+name+'.pkl')
    result = pipe._train_model.predict([value])
else:
    model = AutoModelForSequenceClassification.from_pretrained("model/"+name+"/", num_labels=2,ignore_mismatched_sizes=True)
    if name[-1].isdigit():
        name = name.rstrip(name[-1])
        is4 = True
    tokenizer = AutoTokenizer.from_pretrained(name+"-base-"+para)
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, top_k=1)
    result = pipe(value)

st.write(f'the result of the sentence " {value} " is {toText(result)}')