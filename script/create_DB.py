#!/usr/bin/python3
import pandas as pd
from DataBuilder import DataBuilder

if __name__ == "__main__":
    
    print("How many tweet do you want to extract? (enter a number to multiple by 100)")
    nb = int(input())
    
    db = []
    
    for i in range(1, nb+1):
        dataBuilder = DataBuilder()
        print("100 entries as added to database")
        db.append(dataBuilder.create_db())

    pd.concat(db).to_csv("final_train_tweet.csv", index=False, sep=";")