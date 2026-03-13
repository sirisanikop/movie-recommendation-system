import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# load datasets
movies = pd.read_csv("dataset/movies.csv")
ratings = pd.read_csv("dataset/ratings.csv")

# merge ratings and movie titles
data = pd.merge(ratings, movies, on="movieId")

# create user-movie matrix
movie_matrix = data.pivot_table(
    index="userId",
    columns="title",
    values="rating"
)

# replace missing ratings with 0
movie_matrix = movie_matrix.fillna(0)

# calculate similarity between movies
similarity = cosine_similarity(movie_matrix.T)

def recommend(movie_name):

    if movie_name not in movie_matrix.columns:
        print("Movie not found in dataset.")
        return

    movie_index = movie_matrix.columns.get_loc(movie_name)

    similarity_scores = list(enumerate(similarity[movie_index]))

    sorted_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    print("\nRecommended movies:\n")

    for i in sorted_scores[1:6]:
        print(movie_matrix.columns[i[0]])


movie = input("Enter a movie name: ")

recommend(movie)