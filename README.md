# Sentimental Review [DEPRECATED]

**WARNING ! This project is deprecated, indeed the twitter API has changed, and Imdb has changed its website, so the data is no longer available...**

**Authors** : 

- El boudiri Anasse
- Falfoul Jihad
- Steiner Jan

## Overview

This project consists of analyzing the sentiments of tweets related to movies / series and then assigning them a rating and comparing them to the IMDb rating of the movie, with the aim of comparing the opinion of twitter against IMDb opinions. For this we will use the Twitter API to retrieve the tweets and be able to filter in order to create a labeled Dataset in order to train a NLP model.

## Getting Started

### Prerequisites

#### Linux (Ubuntu 22.04, python 3.10.4)

- Dependencies:

```cmd
>>> cd sentimental-review
>>> pip install pipenv
>>> pipenv install
>>> pipenv shell
>>> python3 -m spacy download en_core_web_sm
>>> streamlit run streamlit/sentimental_analysis.py
```

You can also use the requirements.txt, but to use streamlit, the best is pipenv or conda. More info [here](https://docs.streamlit.io/library/get-started/installation)

#### Windows

- Install [MiniConda3](https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Windows-x86_64.exe).
- Add folder condabin to PATH.
- Make sure the folder where the scripts installed by python is added to the PATH.
- In the command prompt:

```cmd
>>> conda activate base 
>>> pip install -r requirements.txt
>>> python3 -m spacy download en_core_web_sm
>>> streamlit run streamlit/sentimental_analysis.py
```

### Data

Get the Kaggle dataset [here](https://www.kaggle.com/datasets/nisargchodavadiya/imdb-movie-reviews-with-ratings-50k) then run the setup.py script to create the right datasets:

```cmd
>>> cd sentimental-review
>>> python3 scripts/setup.py
```

### API Twitter

To use the Twitter API, we need an access key, these are accessible by creating a Twitter developer account. For more info [here](https://developer.twitter.com/en/docs/twitter-api/getting-started/getting-access-to-the-twitter-api). You can also use the access key that we have made available on Teams.

Then you need to add the access key in the config.yaml.

