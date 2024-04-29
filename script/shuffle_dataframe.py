# Shuffle the dataset

import pandas as pd
import os

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, '../data/imdb_sup.csv')
save_filename = os.path.join(dirname, '../data/imdb_shuffle.csv')


df = pd.read_csv(filename)
df = df.sample(frac=1).reset_index(drop=True)  # shuffle
df.to_csv(save_filename, index=False)
