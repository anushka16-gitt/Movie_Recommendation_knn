import os
import numpy as np
import pandas as pd
from fuzzywuzzy import process


def read_data(folder):
    movies = pd.read_csv(os.path.join(folder, "movies.csv"))
    ratings = pd.read_csv(os.path.join(folder, "ratings.csv"))
    return movies, ratings

# filter out users and movies with too few ratings
def clean_data(ratings, min_user=50, min_movie=10):
    user_count = ratings["userId"].value_counts()
    movie_count = ratings["movieId"].value_counts()

    good_users = user_count[user_count >= min_user].index
    good_movies = movie_count[movie_count >= min_movie].index

    result = ratings[
        (ratings["userId"].isin(good_users)) &
        (ratings["movieId"].isin(good_movies))
    ]
    return result

# create a matrix where rows are movies and columns are users
def make_matrix(ratings, movies):
    matrix = ratings.pivot_table(index="movieId", columns="userId", values="rating").fillna(0)
    names = movies.set_index("movieId")["title"].to_dict()
    title_list = [names.get(mid, "Unknown") for mid in matrix.index]
    return matrix, title_list

def search_movie(name, title_list):
    match = process.extractOne(name, title_list)
    if match and match[1] >= 50:
        found = match[0]
        pos = title_list.index(found)
        return found, pos
    return None, None

def suggest_movies(movie_name, matrix, knn_model, title_list, count=10):
    found_title, pos = search_movie(movie_name, title_list)
    if found_title is None:
        return None, []

    movie_row = matrix.iloc[pos].values.reshape(1, -1)
    dist, idx = knn_model.kneighbors(movie_row, n_neighbors=count + 1)

    suggestions = []
    for i in range(1, len(idx[0])):
        movie_idx = idx[0][i]
        score = round(1 - dist[0][i], 4)
        suggestions.append((title_list[movie_idx], score))

    return found_title, suggestions
