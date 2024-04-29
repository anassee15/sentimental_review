#!/usr/bin/python3
import pandas as pd
import os


def create_label(x):
    if (x > 0 and x <= 2):
        return 0  # very bad
    elif (x > 2 and x <= 4):
        return 1  # bad
    elif (x > 6 and x <= 8):
        return 2  # good
    elif (x > 8 and x <= 10):
        return 3  # very good
    else:
        return -1


dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, '../data/imdb_shuffle.csv')

df = pd.read_csv(filename)
df["Label"] = df["Rating"].apply(create_label)
df.to_csv(filename, index=False)
print("Done")
