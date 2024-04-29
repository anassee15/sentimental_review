#!/usr/bin/python3
# Fine-tune script with pre-trained model from hugging face to create new model
import pandas as pd
import numpy as np
import evaluate
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

from datasets import Dataset, DatasetDict

# data path
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, '../data/imdb_sup.csv')

# split data
df = pd.read_csv(filename)
df = df[:40000]
X = list(df["Review"])
y = list(df["Sentiment"])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0)

# convert pandas dataframe to hugging face dataset
ds = DatasetDict()
df_train = pd.DataFrame({"text": X_train, "label": y_train})
df_test = pd.DataFrame({"text": X_test, "label": y_test})
train = Dataset.from_pandas(df_train)
test = Dataset.from_pandas(df_test)


ds = DatasetDict()

ds["train"] = train
ds["test"] = test

# get the pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = ds.map(tokenize_function, batched=True)

# get the pre-trained model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=5)


# Evaluate method for validation dataset
metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# train new model

training_args = TrainingArguments(
    output_dir="test_trainer", evaluation_strategy="epoch")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

trainer.train()

# save model
trainer.save_model(output_dir="fine-tune-model")
trainer.save_state()
