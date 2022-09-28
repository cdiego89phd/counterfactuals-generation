# import polyjuice
# from typing import Tuple
# from polyjuice.polyjuice import pol
import nltk
# nltk.download()
from nltk.corpus import movie_reviews
# from nltk.corpus import
from sklearn.datasets import load_files
import pandas as pd
import bs4


if __name__ == "__main__":
    print("ciao")

    movie_train = load_files('imdb_pang')

    # Remove HTML from reviews
    reviews = [bs4.BeautifulSoup(r, features="lxml").get_text() for r in movie_train.data]

    # review_text = bs4.BeautifulSoup(review).get_text()
    print(reviews[0])

    d = {"review": reviews,
         "sentiment": movie_train.target}
    df_reviews = pd.DataFrame(data=d)

    # divide df into positive and negative
    seed = 2022

    df_pos = df_reviews[df_reviews["sentiment"] == 1].copy()
    df_neg = df_reviews[df_reviews["sentiment"] == 0].copy()
    print(f"len positive df: {len(df_pos)}")
    print(f"len negative df: {len(df_neg)}")

    # sample balanced train-test split (75-25)%
    test_pos = df_pos.sample(frac=0.25, replace=True, random_state=seed)
    train_pos = df_pos[~df_pos.index.isin(test_pos.index)]
    # train_pos = df_pos.loc[test_pos]
    print(f"len positive test: {len(test_pos)}")
    print(f"len positive train: {len(train_pos)}")

    print("ciao")




