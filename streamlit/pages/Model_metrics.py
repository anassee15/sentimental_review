import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt
import seaborn as sns

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer

from datasets import Dataset,DatasetDict
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay, classification_report
from tfidf import *  
import os




st.markdown("# Model metrics 📊")
st.sidebar.markdown("# Model metrics 📊")

option = st.selectbox('Which model would u like to use ?',('Distilbert uncased', 'Bert cased', 'tfidf ', 'countVectorizer ','Distilbert4 uncased', 'Bert4 cased'))
option = option.lower()
name, para = option.split(' ')

nbSample = st.slider('Number of sample', 0, 10000, 100)

data_path = "data/imdb_shuffle.csv"

df = pd.read_csv(data_path, engine='python', encoding='utf-8', error_bad_lines=False)
df=df[-nbSample:]
df.reset_index(drop=True)

X_pred=list(df["Review"])
y_true = list(df["Sentiment"])
y_true4 = list(df["Label"])



def bert_tokenize_function(examples):
    return tokenizer(examples["text"],padding="max_length", truncation=True)




if os.path.exists('model/'+name+'/'+name+'.pkl'):
    if name == 'tfidf':
        pipe = TfidfPipepline(pkl_name='model/'+name+'/'+name+'.pkl')
    else:
        pipe = CountVectorizerPipepline(pkl_name='model/'+name+'/'+name+'.pkl')
    preds= pipe._train_model.predict(X_pred)
else:
    model = AutoModelForSequenceClassification.from_pretrained("model/"+name+"/", ignore_mismatched_sizes=True)
    if name[-1].isdigit():
        name = name.rstrip(name[-1])
        y_true = y_true4

    ds = DatasetDict()
    df_test=pd.DataFrame({"text":X_pred,"label":y_true})
    test = Dataset.from_pandas(df_test)
    ds["test"] = test

    print(df_test)

    tokenizer = AutoTokenizer.from_pretrained(name+"-base-"+para)
    trainer=Trainer(model=model)
    tokenized_datasets = ds.map(bert_tokenize_function,batched=True)
    prediction=trainer.predict(tokenized_datasets["test"])
    preds = np.argmax(prediction.predictions, axis=-1)








def plot_cm(y_true, y_pred, title):
    ''''
    input y_true-Ground Truth Labels
          y_pred-Predicted Value of Model
          title-What Title to give to the confusion matrix
    
    Draws a Confusion Matrix for better understanding of how the model is working
    
    return None
    
    '''
    
    figsize=(10,10)
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)
    st.pyplot(fig)


st.write(classification_report(y_pred=preds,y_true=y_true,output_dict=True))
plot_cm(y_true=y_true,y_pred=preds,title='confusion matrix of '+name)


